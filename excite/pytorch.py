"""
    @Author Jay Lee
    Some simple utility methods for working with pytorch.
"""
import torch
import torch.nn as nn
import functools
from torchvision.utils import save_image

try:
    from .util import time_it
except:
    from util import time_it

"""
    Decorator section
    ======================
"""


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
                # print(train_fn)
                # print(train_fn.__name__)
                train_fn(model, epoch, *args, **kwargs)

            save_dir = kwargs['save_dir']
            if save_dir is not None:
                torch.save(model.state_dict(), save_dir)

            finished_fn = kwargs['training_finished']
            if callable(finished_fn):
                finished_fn(model)

        return trainer_fn

    return decorate


"""
    End decorator section
    ======================
"""


class Datasets:
    """
        TODO: Create a utility class for handling
        that helps us get both the test and training set data
    """

    def __init__(self):
        pass


class Module(nn.Module):
    # TODO: Create a module that gets rid of
    # boilerplate code, but achieves the same amount of flexibility
    # Try writing more pytorch code so that we get a better idea of some
    # of the pain points and improve on that
    def __init__(self, batch_norm=nn.BatchNorm1d):
        pass

    def _add_batch_norm(self):
        pass


class DAGMM(nn.Module):
    """
        An implementation of the Deep Autoencoding
        Gaussian Mixture Model.
    """

    def __init__(self):
        super().__init__()


class Autoencoder(nn.Module):
    """
        A basic Autoencoder
    """

    def __init__(self, encoder_layers, activation=nn.ReLU(True), batch_norm=None, dropout=nn.Dropout(0.5)):
        super().__init__()
        self.activation = activation
        # All the way up until the last layer
        temp_encoder_layers = []
        for layer in encoder_layers[:-1]:
            temp_encoder_layers.append(nn.Linear(layer[0], layer[1]))
            temp_encoder_layers.append(self.activation)
            if batch_norm is not None:
                temp_encoder_layers.append(batch_norm)
            if dropout is not None:
                temp_encoder_layers.append(dropout)

        temp_encoder_layers.append(nn.Linear(encoder_layers[-1][0], encoder_layers[-1][1]))
        temp_decoder_layers = []
        # Decoder layers is the direct inverse of original input
        for layer in reversed(encoder_layers[1:]):
            temp_decoder_layers.append(nn.Linear(layer[1], layer[0]))
            temp_decoder_layers.append(self.activation)
            if batch_norm is not None:
                temp_decoder_layers.append(batch_norm)
            if dropout is not None:
                temp_decoder_layers.append(dropout)

        temp_decoder_layers.append(nn.Linear(encoder_layers[0][1], encoder_layers[0][0]))

        self.encoder = nn.Sequential(*temp_encoder_layers)
        self.decoder = nn.Sequential(*temp_decoder_layers)

        # Decoders

    def forward(self, X):
        """
            Feed forwarding. Autoencoder learning is unsupervised.
            Reconstruction error used will be measured by euclidean distance
            :param X:
            :return:
        """
        encoder_output = self.encoder(X)
        decoder_output = self.decoder(encoder_output)
        return decoder_output


class GMM(nn.Module):
    """
        A gaussian mixture model class.
        This generic model will be trained using expectation-maximization (EM)
        algorithm
    """

    def __init__(self):
        super().__init__()

    def forward(self, X):
        # TODO: Work on this as soon as I can
        return X


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


def _get_fc_layers(layers, activation):
    """
        :param layers: The specification of the layer. Each item contains
        an input dimension and output dimension. E.g. (128, 64) ->
        input size of 128, output neuron of 64.
        :param nn_layer:
        :return:
    """
    temp_layers = []
    for layer in layers[:-1]:
        temp_layers.append(nn.Linear(layer[0], layer[1]))
        temp_layers.append(activation)

    final_layer = layers[len(layers) - 1]
    temp_layers.append(nn.Linear(final_layer[0], final_layer[1]))
    return temp_layers


class Discriminator(nn.Module):
    def __init__(self, layers, activation=nn.LeakyReLU):
        super().__init__()
        layers = _get_fc_layers(layers)
        # Final layer is a tanh layer
        layers.append(nn.Tanh())
        self.layers = nn.Sequential()

    def forward(self, X):
        pass


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        pass


class GAN(nn.Module):
    """
        Simple Generative Adversarial Network (By Goodfellow, 2014).
        
    """
    def __init__(self, gen_layers, disc_layers, activation=nn.ReLU):
        super().__init__()
        self.activation = activation
        self.generator = Generator(gen_layers)
        self.discriminator = Discriminator(disc_layers)

    def forward(self, X):
        """
            Forward pass of a GAN. Given some random noise ~ N(0, 1)
            generate data that is likely from a similar distribution
            to the input data
            :param X:
            :return:
        """
        generated_image = self.generator(X)

        # How do we pass both images to the discriminator
        # For evaluation?
        discriminated_output = self.discriminator(generated_image)


        return generated_image

    def loss_function(self, X):
        """
            Loss function of the GAN
            :param X:
            :return:
        """
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


class ExtendedMLP(MLP):
    # TODO: Work on this in the near future
    def forward(self, X):
        # Just extend and modify behavior here if you need a specific implementation
        return "teemo"


# Place "time_it" on top of train to get the the amount of time it takes to complete the entire
# training phase
@time_it()
@train(epochs=10, lr=0.0001,
       save_dir="../data/model/teemo.pt",
       training_finished=lambda x: print(f"Finished training the following model: {x}"))
# If you want to measure the amount of time per epoch, place @time_it
# Before @train
def train_mlp(model, epoch, *args, **kwargs):
    """
        Simple example of training a model using the
        @train decorator.
        :param model: The model that we will be training
        :param epoch: The current epoch
        :param args: Extra arguments passed
        :param kwargs: Key word args. Can be used to override the values
        specified in @train or the default values to assign hyperparameters
        dynamically
        :return:
    """
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


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


@train(save_dir="../data/model/trained_autoencoder.pt", epochs=10)
def train_autoencoder(model, epoch, *args, **kwargs):
    data_loader = kwargs['data']
    CUDA = kwargs['cuda']
    optimizer = kwargs['optimizer']
    # print(model.encoder)
    # print(model.decoder)

    for data in data_loader:
        img, _ = data
        if CUDA:
            img = img.cuda()

        img = img.view(img.size(0), -1)

        y_pred = model(img)
        loss = kwargs['criterion'](img, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch, kwargs['epochs'], loss.item()))
    if epoch % 2 == 0:
        pic = to_img(img.data)
        save_image(pic, '../data/image_{}.png'.format(epoch))



@train(epochs=100)
def train_GAN(model, epoch, *arg, **kwargs):
    data_loader = kwargs['data']
    CUDA = kwargs['cuda']
    optimizer = kwargs['optimizer']

    for data in data_loader:
        img, _ = data

        # Generate random noise
        random_noise = torch.empty(img.size(0)).normal_(mean=0, std=1)

        generated_images = model(random_noise)

        print(random_noise)
        break

    # print('epoch [{}/{}], loss:{:.4f}'
    #       .format(epoch, kwargs['epochs'], loss.item()))


def mnist_mlp_example(data_loader):
    # MNIST MLP Example
    mlp = MLP([
        (784, 400),
        (400, 400),
        (400, 10)
    ])

    train_mlp(mlp, data=data_loader, lr=0.01, criterion=nn.CrossEntropyLoss())


def mnist_autoencoder_example(autoencoder, data_loader):
    # MNIST Autoencoder example
    autoencoder = Autoencoder([
        [784, 512],
        [512, 256],
        [256, 64],
        [64, 3]
    ])

    train_autoencoder(autoencoder.cuda(), cuda=torch.cuda.is_available(), data=data_loader, lr=0.001,
                      criterion=nn.MSELoss())


def mnist_GAN(data_loader):
    gan = GAN([
        # Generator
        (2, 256),
        (256, 256),
        (256, 784)
    ],
        # Discriminator
        # Output: 1 if from actual dataset
        # Otherwise, 0
        [
            (784, 256),
            (256, 256),
            (256, 1)
        ]
    )

    train_GAN(gan.cuda(), cuda=torch.cuda.is_available(), data=data_loader, lr=0.001,
                      criterion=nn.MSELoss())


if __name__ == "__main__":
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data.dataloader import DataLoader

    training_data = datasets.MNIST(root="../data/", download=True, train=True, transform=transforms.ToTensor())
    data_loader = DataLoader(dataset=training_data, shuffle=True, batch_size=128)

    mnist_GAN(data_loader)

