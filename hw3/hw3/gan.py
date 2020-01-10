from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        self.first_channels = 3
        self.last_channels = 512

        downscaling1 = nn.Sequential(
            nn.Conv2d(self.first_channels, 128, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128)
        )
        downscaling2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256)
        )
        downscaling3 = nn.Sequential(
            nn.Conv2d(256, self.last_channels, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(self.last_channels)
        )
        self.downscaler = nn.Sequential(downscaling1, downscaling2, downscaling3)
        output_size = (in_size[1] // 8, in_size[2] // 8)
        self.final_layer = nn.Linear(self.last_channels * output_size[0] * output_size[1], 1)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        downscaled = self.downscaler(x)
        reshaped = downscaled.view([x.shape[0], -1])
        y = self.final_layer(reshaped)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        self.first_featuremap_size = featuremap_size
        self.out_channels = out_channels
        self.first_channels = 1024
        self.output_size = 64

        self.projection_layer = nn.Sequential(
            nn.Linear(z_dim, self.first_channels * featuremap_size * featuremap_size),
        )

        upscaling1 = nn.Sequential(
            nn.ConvTranspose2d(self.first_channels, 512, 4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        upscaling2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        upscaling3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        upscaling4 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        self.upscaler = nn.Sequential(upscaling1, upscaling2, upscaling3, upscaling4)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        torch.autograd.set_grad_enabled(with_grad)

        input_batch = torch.randn(n, self.z_dim, requires_grad=with_grad).to(device)
        samples = self.forward(input_batch)

        if not with_grad:
            torch.autograd.set_grad_enabled(True)

        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        projected = self.projection_layer(z)
        reshaped = projected.view([z.shape[0], self.first_channels,
                                   self.first_featuremap_size, self.first_featuremap_size])
        x = self.upscaler(reshaped)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    device = y_data.device
    loss_fn = nn.BCEWithLogitsLoss()
    data_noise = torch.rand(*y_data.shape) * label_noise - (label_noise / 2)
    generated_noise = torch.rand(*y_data.shape) * label_noise - (label_noise / 2)

    loss_data = loss_fn(y_data, (data_noise + data_label).to(device))
    loss_generated = loss_fn(y_generated, (generated_noise + (1 - data_label)).to(device))
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    device = y_generated.device
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(y_generated, torch.full_like(y_generated, data_label, device=device))
    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()

    real_batch = x_data
    generated_batch = gen_model.sample(len(real_batch), with_grad=True)

    y_data = dsc_model(real_batch)
    y_generated = dsc_model(generated_batch.detach())

    dsc_loss = dsc_loss_fn(y_data, y_generated)
    dsc_loss.backward()

    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()

    y_generated = dsc_model(generated_batch)

    gen_loss = gen_loss_fn(y_generated)
    gen_loss.backward()

    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f'{checkpoint_file}.pt'

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    epoch = len(gen_losses)

    saved = True
    if len(gen_losses) >= 2:
        if gen_losses[-1] > gen_losses[-2]:
            saved = False

    if saved and checkpoint_file is not None:
        saved_state = gen_model
        torch.save(saved_state, checkpoint_file)
        print(f'*** Saved checkpoint {checkpoint_file} '
              f'at epoch {epoch}')
    # ========================

    return saved
