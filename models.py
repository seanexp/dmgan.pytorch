import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn


class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()

        self.func = func

    def forward(self, x):
        return self.func(x)


class Generator(nn.Module):
    def __init__(self, nz, n_gen, n_div):
        super(Generator, self).__init__()

        self.nz = nz
        self.n_gen = n_gen
        self.n_div = n_div

    def forward(self, z, g_idx):
        batch_size = z.size(0)

        images = torch.stack([self.gens[i](z) for i in range(self.n_gen)], dim=1)
        images = images[torch.arange(batch_size), g_idx]

        return images


class Discriminator(nn.Module):
    def __init__(self, n_gen):
        super(Discriminator, self).__init__()

    def forward(self, input):
        x = self.latent(input)
        score = self.score(x)
        posterior = self.posterior(x)
        return score, posterior


class MNISTGenerator(Generator):
    def __init__(self, *args, **kwargs):
        super(MNISTGenerator, self).__init__(*args, **kwargs)

        n_div = self.n_div
        self.gens = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(self.nz, 1024 // n_div),
                            nn.LeakyReLU(0.2),
                            nn.Linear(1024 // n_div, 128 // n_div * 7 * 7),
                            nn.LeakyReLU(0.2),
                            Lambda(lambda x: x.view(-1, 128 // n_div, 7, 7)),           # B X 32 X 7 X 7
                            nn.ConvTranspose2d(128 // n_div, 64 // n_div, 4, 2, 1),     # B X 16 X 14 X 14
                            nn.LeakyReLU(0.2),
                            nn.ConvTranspose2d(64 // n_div, 32 // n_div, 4, 2, 1),      # B X 8 X 28 X 28
                            nn.LeakyReLU(0.2),
                            nn.ConvTranspose2d(32 // n_div, 1, 3, 1, 1),                # B X 1 X 28 X 28
                            nn.Tanh(),
                            )
                        for i in range(self.n_gen)
                    ])


class MNISTShareGenerator(Generator):
    def __init__(self, *args, **kwargs):
        super(MNISTShareGenerator, self).__init__(*args, **kwargs)

        self.common = nn.Sequential(
                            nn.Linear(self.nz, 1024),
                            nn.LeakyReLU(0.2),
                            nn.Linear(1024, 128 * 7 * 7),
                            nn.LeakyReLU(0.2),
                            Lambda(lambda x: x.view(-1, 128, 7, 7)),  # B X 32 X 7 X 7
                            nn.ConvTranspose2d(128, 64, 4, 2, 1),     # B X 16 X 14 X 14
                            nn.LeakyReLU(0.2),
                    )
        n_div = self.n_div
        self.gens = nn.ModuleList([
                        nn.Sequential(
                            nn.ConvTranspose2d(64, 32 // n_div, 4, 2, 1),      # B X 8 X 28 X 28
                            nn.LeakyReLU(0.2),
                            nn.ConvTranspose2d(32 // n_div, 1, 3, 1, 1),                # B X 1 X 28 X 28
                            nn.Tanh(),
                            )
                        for i in range(self.n_gen)
                    ])

    def forward(self, z, g_idx):
        batch_size = z.size(0)

        latent = self.common(z)
        images = torch.stack([self.gens[i](latent) for i in range(self.n_gen)], dim=1)
        images = images[torch.arange(batch_size), g_idx]

        return images


class MNISTDiscriminator(Discriminator):
    def __init__(self, n_gen):
        super(MNISTDiscriminator, self).__init__(n_gen)

        self.latent = nn.Sequential(
                sn(nn.Conv2d(1, 64, 4, 2, 1)),                  # B X 64 X 14 X 14
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                sn(nn.Conv2d(64, 128, 4, 2, 1)),                # B X 128 X 7 X 7
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                Lambda(lambda x: x.view(-1, 128 * 7 * 7)),
                )

        self.score = nn.Sequential(
                sn(nn.Linear(128 * 7 * 7, 1024)),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2),
                sn(nn.Linear(1024, 1)),
                )

        self.posterior = nn.Sequential(
                nn.Linear(128 * 7 * 7, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, n_gen),
                )


class ChairsGenerator(Generator):
    def __init__(self, *args, **kwargs):
        super(ChairsGenerator, self).__init__(*args, **kwargs)

        n_div = self.n_div
        self.gens = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(self.nz, 256 // n_div),
                            nn.LeakyReLU(0.2),
                            nn.Linear(256 // n_div, 128 // n_div * 8 * 8),
                            nn.LeakyReLU(0.2),
                            Lambda(lambda x: x.view(-1, 128 // n_div, 8, 8)),           # B X 32 X 8 X 8
                            nn.ConvTranspose2d(128 // n_div, 64 // n_div, 4, 2, 1),     # B X 16 X 16 X 16
                            nn.LeakyReLU(0.2),
                            nn.ConvTranspose2d(64 // n_div, 32 // n_div, 4, 2, 1),      # B X 8 X 32 X 32
                            nn.LeakyReLU(0.2),
                            nn.ConvTranspose2d(32 // n_div, 16 // n_div, 4, 2, 1),      # B X 4 X 64 X 64
                            nn.LeakyReLU(0.2),
                            nn.ConvTranspose2d(16 // n_div, 1, 3, 1, 1),                # B X 1 X 64 X 64
                            nn.Tanh(),
                            )
                        for i in range(self.n_gen)
                    ])


class ChairsShareGenerator(Generator):
    def __init__(self, *args, **kwargs):
        super(ChairsShareGenerator, self).__init__(*args, **kwargs)

        self.common = nn.Sequential(
                            nn.Linear(self.nz, 256),
                            nn.BatchNorm1d(256),
                            nn.LeakyReLU(0.2),
                            nn.Linear(256, 128 * 8 * 8),
                            nn.BatchNorm1d(128 * 8 * 8),
                            nn.LeakyReLU(0.2),
                            Lambda(lambda x: x.view(-1, 128, 8, 8)),           # B X 32 X 8 X 8
                            nn.ConvTranspose2d(128, 64, 4, 2, 1),     # B X 16 X 16 X 16
                            nn.BatchNorm2d(64),
                            nn.LeakyReLU(0.2),
                            nn.ConvTranspose2d(64, 32 // n_div, 4, 2, 1),      # B X 8 X 32 X 32
                )

        n_div = self.n_div
        self.gens = nn.ModuleList([
                        nn.Sequential(
                            nn.BatchNorm2d(32 // n_div),
                            nn.LeakyReLU(0.2),
                            nn.ConvTranspose2d(32 // n_div, 16 // n_div, 4, 2, 1),      # B X 4 X 64 X 64
                            nn.BatchNorm2d(16 // n_div),
                            nn.LeakyReLU(0.2),
                            nn.ConvTranspose2d(16 // n_div, 1, 3, 1, 1),                # B X 1 X 64 X 64
                            nn.Tanh(),
                            )
                        for i in range(self.n_gen)
                    ])

    def forward(self, z, g_idx):
        batch_size = z.size(0)

        latent = self.common(z)
        images = torch.stack([self.gens[i](latent) for i in range(self.n_gen)], dim=1)
        images = images[torch.arange(batch_size), g_idx]

        return images


class ChairsDiscriminator(Discriminator):
    def __init__(self, n_gen):
        super(ChairsDiscriminator, self).__init__(n_gen)

        self.latent = nn.Sequential(
                sn(nn.Conv2d(1, 64, 4, 2, 1)),                  # B X 64 X 32 X 32
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                sn(nn.Conv2d(64, 128, 4, 2, 1)),                # B X 128 X 16 X 16
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                sn(nn.Conv2d(128, 128, 4, 2, 1)),                # B X 128 X 8 X 8
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                Lambda(lambda x: x.view(-1, 128 * 8 * 8)),
                )

        self.score = nn.Sequential(
                sn(nn.Linear(128 * 8 * 8, 1024)),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2),
                sn(nn.Linear(1024, 1)),
                )

        self.posterior = nn.Sequential(
                nn.Linear(128 * 8 * 8, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, n_gen),
                )
