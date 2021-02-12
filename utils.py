import numpy as np
import torch
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt


def entropy(x: torch.Tensor) -> torch.Tensor:
    x = x / x.sum(dim=1, keepdim=True)
    return -(x.clamp(min=0.001).log()*x).sum(dim=1)


def prediction_probability(x: torch.Tensor) -> torch.Tensor:
    proba, pred = x.max(dim=1)
    return proba

def ellipse(x, y, color, label, n_std=1, linestyle='solid'):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    result = Ellipse((0, 0),
                        width=ell_radius_x * 2,
                        height=ell_radius_y * 2,
                        edgecolor=color,
                        facecolor="None",
                        linestyle=linestyle,
                        linewidth=2.5,
                        alpha=0.5,
                        label=label)


    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D()
    tansf = transf.rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    return result, transf


def histogram_report(predictions, uncertainty, ground_truth, title):
    correct = (predictions.max(dim=1)[1] == ground_truth)
    print("acc = ", correct.float().mean().item())
    correct_uncertainty = uncertainty[correct].numpy()
    incorrect_uncertainty = uncertainty[~correct].numpy()
    xmin = min(correct_uncertainty.min(), incorrect_uncertainty.min())
    xmax = max(correct_uncertainty.max(), incorrect_uncertainty.max())
    # now we plot the two distributions
    plt.clf()
    _, bins, _ = plt.hist(
                    incorrect_uncertainty,
                    bins=50,
                    range=[xmin, xmax],
                    label='incorrect',
                    density=True,
                    color='red')
    plt.hist(correct_uncertainty,
                    bins=bins,
                    alpha=0.5,
                    label='correct',
                    density=True,
                    color='blue')
    plt.legend()
    plt.yticks([])
    plt.xlabel("uncertainty measure")
    if title:
        plt.title(title)
    plt.savefig(title.replace(" ", "_").replace("'",""), dpi=80)
