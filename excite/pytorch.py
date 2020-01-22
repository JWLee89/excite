"""
    @Author Jay Lee
    Some simple utility methods for working with pytorch.
"""
import torch
import torch.nn as nn
import functools


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


def _add_if_not_exist(args_dict, kwargs):
    """
        Add to keyword args if it does not exist
        :param args_dict:
        :param kwargs:
        :return: None
    """
    for key, value in args_dict.items():
        if key not in kwargs:
            kwargs[key] = value


def train(epochs=500, batch_size=128,
          optimizer=torch.optim.Adam, lr=0.001,
          criterion=None,
          used_gpu=None,
          patience=None, save_dir=None,
          training_finished=None):
    """
        A convenient decorator for annotating a training function.
        Used to reduce as many boilerplate code as possible.

        :param epochs: The number of epochs to run
        :param criterion:
        :param used_gpu: The GPUs that we wish to use
        :param optimizer:
        :param lr: Learning rate
        :param patience: If early stopping is used, set this to an integer value
        :param save_dir: If specified, model will be saved to this particular directory
        :return:
    """
    decorator_scope = locals()

    def decorate(train_fn):
        @functools.wraps(train_fn)
        def trainer_fn(model, *args, **kwargs):
            _add_if_not_exist(decorator_scope, kwargs)
            kwargs['optimizer'] = kwargs['optimizer'](model.parameters(), lr=lr)
            for epoch in range(1, kwargs['epochs'] + 1):
                accuracy, loss = train_fn(model, epoch, *args, **kwargs)

            save_dir = kwargs['save_dir']
            if save_dir is not None:
                print(f"saving to directory: {save_dir}")

            finished_fn = kwargs['training_finished']
            if callable(finished_fn):
                finished_fn(model)

        return trainer_fn

    return decorate


class ExtendedMLP(MLP):
    def forward(self, X):
        # Just extend and modify behavior here if you need a specific implementation
        return "teemo"


@train(epochs=10, lr=0.0001, save_dir="teemo", training_finished=lambda x: print("training has finished yaye!"))
def train_mnist(model, epoch, *args, **kwargs):

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
    data_loader = kwargs['data']

    total_correct = 0
    total = 0
    losses = []
    optimizer = kwargs['optimizer']

    for step, (x_train, y_train) in enumerate(data_loader):
        x_train = x_train.reshape(-1, 28 * 28)

        if cuda:
            x_train = x_train.cuda()
            y_train = y_train.cuda()

        optimizer.zero_grad()

        y_pred = model(x_train)
        _, prediction = torch.max(y_pred, 1)

        total += y_pred.size(0)

        total_correct += (prediction == y_train).sum().item()

        loss = kwargs['criterion'](y_pred, y_train)
        losses.append(loss)

        # Train
        loss.backward()
        optimizer.step()

        step += 1

    accuracy = total_correct / total
    print(f"Epoch {epoch}/{kwargs['epochs']}. Accuracy: {accuracy:.3f}")

    return accuracy, losses


if __name__ == "__main__":
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    training_data = datasets.MNIST(root="../data/", download=True, train=True, transform=transforms.ToTensor())

    mlp = MLP([
        (784, 400),
        (400, 400),
        (400, 10)
    ])
    from torch.utils.data.dataloader import DataLoader

    data_loader = DataLoader(dataset=training_data, shuffle=True, batch_size=128)
    train_mnist(mlp, data=data_loader, lr=0.1, criterion=nn.CrossEntropyLoss())
