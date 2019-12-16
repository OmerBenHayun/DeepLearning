import torch
import itertools as it
import torch.nn as nn


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes: int, channels: list,
                 pool_every: int, hidden_dims: list):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ReLU)*P -> MaxPool]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs should exist at the end, without a MaxPool after them.
        # ====== YOUR CODE: ======
        channels = [in_channels, *self.channels]
        for i, (conv_in_channels, conv_out_channels) in enumerate(zip(channels, channels[1:])):
            layers.append(nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            if (i + 1) % self.pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        #  (Linear -> ReLU)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        in_features_divider = len(self.channels) // self.pool_every
        in_features_h = in_h // (2 ** in_features_divider)
        in_features_w = in_w // (2 ** in_features_divider)
        in_features_channels = self.channels[-1]
        dims = [int(in_features_h * in_features_w * in_features_channels), *self.hidden_dims]
        for h1, h2 in zip(dims, dims[1:]):
            layers.append(nn.Linear(h1, h2))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(h2, self.out_classes))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor.forward(x)
        features = features.view(features.shape[0], -1)
        out = self.classifier(features)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list,
                 batchnorm=False, dropout=0.):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
        convolution in the block. The length determines the number of
        convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
        be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
        convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
        Zero means don't apply dropout.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  the main_path, which should contain the convolution, dropout,
        #  batchnorm, relu sequences, and the shortcut_path which should
        #  represent the skip-connection.
        #  Use convolutions which preserve the spatial extent of the input.
        #  For simplicity of implementation, we'll assume kernel sizes are odd.
        # ====== YOUR CODE: ======
        channels = (in_channels, *channels)
        main_path_layers = []
        for (h1, h2, size) in zip(channels, channels[1:-1], kernel_sizes):
            main_path_layers.append(nn.Conv2d(h1, h2, size, padding=size // 2))
            if dropout > 0:
                main_path_layers.append(nn.Dropout2d(dropout))
            if batchnorm:
                main_path_layers.append(nn.BatchNorm2d(h2))
            main_path_layers.append(nn.ReLU())
        main_path_layers.append(nn.Conv2d(channels[-2], channels[-1], kernel_sizes[-1], padding=kernel_sizes[-1] // 2))
        self.main_path = nn.Sequential(*main_path_layers)

        skip_layers = []
        if channels[0] != channels[-1]:
            skip_layers.append(nn.Conv2d(channels[0], channels[-1], 1, bias=False))
        self.shortcut_path = nn.Sequential(*skip_layers)

        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ReLU)*P -> MaxPool]*(N/P)
        #   \------- SKIP ------/
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs (with a skip over them) should exist at the end,
        #  without a MaxPool after them.
        # ====== YOUR CODE: ======
        channels = [in_channels, *self.channels]
        kernel_sizes = [3] * self.pool_every
        for i in range(0, len(self.channels) - self.pool_every + 1, self.pool_every):
            layers.append(ResidualBlock(channels[i], channels[i + 1:i + self.pool_every + 1], kernel_sizes))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        remainder = len(self.channels) % self.pool_every
        if remainder:
            layers.append(ResidualBlock(channels[-remainder - 1], channels[-remainder:], kernel_sizes[:remainder]))
        # ========================
        seq = nn.Sequential(*layers)
        return seq


class InceptionBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, batchnorm=False, dropout=0.):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: Number of output channels for the entire block
        :param batchnorm: True/False whether to apply BatchNorm between
        convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
        Zero means don't apply dropout.
        """
        super().__init__()

        self.path1, self.path2, self.pool_path, self.shortcut_path = None, None, None, None

        path1_layers = []
        path1_layers.append(nn.Conv2d(in_channels, out_channels, 1))
        if batchnorm:
            path1_layers.append(nn.BatchNorm2d(out_channels))
        path1_layers.append(nn.ReLU())
        if dropout > 0:
            path1_layers.append(nn.Dropout2d(dropout))
        path1_layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        if batchnorm:
            path1_layers.append(nn.BatchNorm2d(out_channels))
        path1_layers.append(nn.ReLU())
        if dropout > 0:
            path1_layers.append(nn.Dropout2d(dropout))
        self.path1 = nn.Sequential(*path1_layers)

        path2_layers = []
        path2_layers.append(nn.Conv2d(in_channels, out_channels, 1))
        if batchnorm:
            path2_layers.append(nn.BatchNorm2d(out_channels))
        path2_layers.append(nn.ReLU())
        if dropout > 0:
            path2_layers.append(nn.Dropout2d(dropout))
        path2_layers.append(nn.Conv2d(out_channels, out_channels, 5, padding=2))
        if batchnorm:
            path2_layers.append(nn.BatchNorm2d(out_channels))
        path2_layers.append(nn.ReLU())
        if dropout > 0:
            path2_layers.append(nn.Dropout2d(dropout))
        self.path2 = nn.Sequential(*path2_layers)

        pool_layers = []
        pool_layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        pool_layers.append(nn.Conv2d(in_channels, out_channels, 1, bias=False))
        self.pool_path = nn.Sequential(*pool_layers)

        shortcut_path_layers = []
        if in_channels != out_channels:
            shortcut_path_layers.append(nn.Conv2d(in_channels, out_channels, 1, bias=False))
        self.shortcut_path = nn.Sequential(*shortcut_path_layers)

    def forward(self, x):
        out = self.path1(x)
        out += self.path2(x)
        out += self.pool_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        dropout = 0.1
        num_layers = len(self.channels)
        channels = [in_channels, *self.channels]
        for i, (h1, h2) in enumerate(zip(channels, channels[1:])):
            layers.append(InceptionBlock(h1, h2, batchnorm=True, dropout=dropout))
            if (i + 1) % self.pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            dropout += 0.3 / num_layers  # dropout will reach 0.4 at the end

        seq = nn.Sequential(*layers)
        return seq
