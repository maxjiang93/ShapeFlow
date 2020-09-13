import torch
import torch.nn as nn


class NoNorm(nn.Module):
    def __init__(self, layers):
        super(NoNorm, self).__init__()

    def forward(self, x):
        return x


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "rmsprop": torch.optim.RMSprop,
}

LOSSES = {
    "l1": torch.nn.L1Loss(),
    "l2": torch.nn.MSELoss(),
    "huber": torch.nn.SmoothL1Loss(),
}

REDUCTIONS = {
    "mean": lambda x: torch.mean(x, axis=-1),
    "max": lambda x: torch.max(x, axis=-1)[0],
    "min": lambda x: torch.min(x, axis=-1)[0],
    "sum": lambda x: torch.sum(x, axis=-1),
}


NORMTYPE = {
    "batchnorm": nn.BatchNorm1d,
    "instancenorm": nn.InstanceNorm1d,
    "none": NoNorm,
}

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x ** 2),
    "identity": Lambda(lambda x: x),
    "leakyrelu": nn.LeakyReLU(),
    "tanh10x": Lambda(lambda x: torch.tanh(10 * x)),
}
