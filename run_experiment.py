#!/usr/bin/env python3

import os
from tempfile import gettempdir
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import make_processor, Frozen, Ensemble, Classifier
from utils import ellipse, prediction_probability, histogram_report

# data
tr = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.15], std=[0.3]),
])
prefix = os.path.join(gettempdir(), "mnist_")
train_set = datasets.MNIST(prefix+"training", download=True, train=True, transform=tr)
test_set = datasets.MNIST(prefix+"testing", download=True, train=False, transform=tr)

def load(loader, digits):
    digit_set = set(digits)
    min_digit = min(digits)
    for images, labels in loader:
        idx = [i for i, l in enumerate(labels) if l.item() in digit_set]
        if len(idx) <= 2:
            continue  # not worth it
        y_true = labels[idx] - min_digit
        onehot = torch.zeros(len(idx), len(digits))
        onehot[np.arange(len(idx)), y_true] = 1
        yield images[idx, :, :, :].view(len(idx), -1), y_true, onehot


# model instantiation
dim = 28 ** 2
latent_dim = 2
digits1 = [0, 1, 2, 3, 4]
digits2 = [5, 6, 7, 8, 9]
head1 = nn.Linear(latent_dim, len(digits1))
head2 = nn.Linear(latent_dim, len(digits2))
frozenhead1 = Frozen(head1)
frozenhead2 = Frozen(head2)
processors1 = [make_processor(dim, latent_dim) for _ in range(4)]
processors2 = [
    make_processor(dim, latent_dim),
    make_processor(dim, latent_dim),
    processors1[2],  # those two shared processors will be interfered with
    processors1[3]
]
classifiers1 = [
    Classifier(processors1[0], head1), # head1 will be trained with 1st processor
    Classifier(processors1[1], frozenhead1),
    Classifier(processors1[2], frozenhead1),
    Classifier(processors1[3], frozenhead1)
]
classifiers2 = [
    Classifier(processors2[0], head2),
    Classifier(processors2[1], frozenhead2),
    Classifier(processors2[2], frozenhead2),
    Classifier(processors2[3], frozenhead2)
]

# testing sets
loader = torch.utils.data.DataLoader(test_set, batch_size=2**31, shuffle=False)
for test_images1, test_labels1, _ in load(loader, digits1):
    pass
loader = torch.utils.data.DataLoader(test_set, batch_size=2**31, shuffle=False)
for test_images2, test_labels2, _ in load(loader, digits2):
    pass

# those two ensembles are utilized for predictions, not training
ensemble1 = Ensemble(processors1, frozenhead1)
ensemble2 = Ensemble(processors2, frozenhead2)

# training
loss_func = nn.BCEWithLogitsLoss()
rounds = []
for digits, classifiers in [(digits1, classifiers1), (digits2, classifiers2)]:
    optimizers = [
        torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        for model in classifiers
    ]
    for epoch in tqdm(range(3), desc="digits %s" % digits, total=3):
        loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
        for images, _, y_true in load(loader, digits):
            for optimizer, model in zip(optimizers, classifiers):
                y_pred = model(images)
                loss = loss_func(y_pred, y_true)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # latent representation of the first set of digits
    shared_processor = processors1[2]
    with torch.no_grad():
        rounds.append(shared_processor(test_images1).numpy())

    with torch.no_grad():
        # running the ensemble on the first set of digits
        title_suffix = "after training on %s" % "_".join(map(str, digits))
        y_pred1, uncertainty = ensemble1(test_images1)
        histogram_report(y_pred1, uncertainty, test_labels1,
            "standard deviation as uncertainty measure "+title_suffix)
        # run a perturbed classifier on the same digits
        y_pred1 = classifiers1[2](test_images1).sigmoid()
        uncertainty = 1 - prediction_probability(y_pred1)
        histogram_report(y_pred1, uncertainty, test_labels1,
            "result's probability as certainty measure "+title_suffix)

# post process the latent representation so they can be shown with matplotlib
plt.clf()
fig, ax = plt.subplots()
labels = test_labels1.numpy()
for digit, color in zip(digits1, ["blue", "red", "green", "orange", "black"]):
    values = rounds[0][labels == digit, :]
    x, y = values[:, 0], values[:, 1]
    shape, transform = ellipse(x, y, color=color, label=str(digit))
    shape.set_transform(transform + ax.transData)
    ax.add_patch(shape)

values = rounds[1][labels == 4, :]
x, y = values[:, 0], values[:, 1]
shape, transf = ellipse(x, y, color="black", linestyle='dashed', label="4 at t+1")
shape.set_transform(transf + ax.transData)
ax.add_patch(shape)
ax.set_aspect('equal')
ax.autoscale(True)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.title("Latent representation of MNIST digits")
plt.legend()
plt.show()
