"""
    @Author Jay Lee
    Some simple utility methods for working with pytorch.
"""
import torch
import torch.nn as nn


class Module(nn.Module):
    pass


class HyperparameterSearcher():
    """
        Class for running hyperparameter search.
        To be used for finding an ideal hyperparameter value for the
        given model and dataset.
    """
    def __init__(self, do_after=None):
        self.combination_dict = {}

    def __call__(self, *args, **kwargs):
        pass


class MLP(nn.Module):
    def __init__(self, layers, activation=nn.RReLU, dropout_rate=0.5, do_batch_normalization=True):
        """

        :param layers:
        :param activation:
        :param dropout_rate:
        :param do_batch_normalization:
        """
        super().__init__()
        self.activation = activation
        temp_layers = []
        for layer in layers[:-1]:
            temp_layers.append(nn.Linear(layer[0], layer[1]))
            temp_layers.append(self.activation())
            if do_batch_normalization:
                temp_layers.append(nn.BatchNorm1d(layer[1]))
            if dropout_rate > 0:
                temp_layers.append(nn.Dropout(dropout_rate))

        temp_layers.append(nn.Linear(layers[-1][0], layers[-1][1]))

        self.layers = nn.Sequential(*temp_layers)

    def forward(self, X):
        output = self.layers(X)
        return output


def print_predicate():
    pass


def train(epochs, criterion, optimizer=torch.optim.Adam, lr=0.001, patience=10, save_dir=None):
    for epoch in epochs:
        pass


class ExtendedMLP(MLP):
    def forward(self, X):
        # Just extend and modify behavior here if you need a specific implementation
        return "teemo"



@deco
def target(teemo):
    print("running target(): ", teemo)


if __name__ == "__main__":
    print("teemo EEEEEEEEEE")
    mlp = ExtendedMLP([
        (784, 400)
    ])