import torch
import torch.nn as nn


def make_processor(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, 128),
        nn.LeakyReLU(0.2),
        nn.Linear(128, 32),
        nn.LeakyReLU(0.2),
        nn.Linear(32, out_dim)
    )

def freeze(module: nn.Module, state: bool=True) -> None:
    for param in module.parameters():
        param.requires_grad_(not state)


class Frozen(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super(Frozen, self).__init__()
        self._module = module

    def forward(self, x):
        freeze(self._module, True)
        result = self._module(x)
        freeze(self._module, False)
        return result


class Classifier(nn.Module):
    def __init__(self, processor: nn.Module, head: nn.Module) -> None:
        super(Classifier, self).__init__()
        self._processor = processor
        self._decision = head

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self._decision(self._processor(img))


class Ensemble(nn.Module):
    def __init__(self, processors: list, classifier) -> None:
        super(Ensemble, self).__init__()
        self._processors = processors  # unregistered parameters
        self._classifier = classifier

    def forward(self, img: torch.Tensor) -> tuple:
        processed = torch.cat([
            p(img).unsqueeze(0) for p in self._processors
        ], dim=0)
        avg = processed.mean(dim=0)
        std = processed.std(dim=0).sum(dim=1)
        y_pred = torch.sigmoid(self._classifier(avg))

        return y_pred, std
