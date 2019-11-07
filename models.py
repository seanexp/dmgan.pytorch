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
    def __init__(self, nz, n_gen):
        super(Generator, self).__init__()

        n_div = 4

        self.gens = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(nz, 1024 // n_div),
                            nn.LeakyReLU(0.2),
                            nn.Linear(1024 // n_div, 128 // n_div * 7 * 7),
                            nn.LeakyReLU(0.2),
                            Lambda(lambda x: x.view(-1, 128 // n_div, 7, 7)),           # B X 32 X 7 X 7
                            nn.ConvTranspose2d(128 // n_div, 64 // n_div, 4, 2, 1),     # B X 32 X 14 X 14
                            nn.LeakyReLU(0.2),
                            nn.ConvTranspose2d(64 // n_div, 32 // n_div, 4, 2, 1),     # B X 32 X 28 X 28
                            nn.LeakyReLU(0.2),
                            nn.ConvTranspose2d(32 // n_div, 1, 3, 1, 1),               # B X 1 X 28 X 28
                            nn.Tanh(),
                            )
                        for i in range(n_gen)
                    ])

        self.nz = nz
        self.n_gen = n_gen

    def forward(self, z, g_idx):
        batch_size = z.size(0)

        images = torch.stack([self.gens[i](z) for i in range(self.n_gen)], dim=1)
        images = images[torch.arange(batch_size), g_idx]

        return images




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                sn(nn.Conv2d(1, 64, 4, 2, 1)),                  # B X 64 X 14 X 14
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                sn(nn.Conv2d(64, 128, 4, 2, 1)),                # B X 128 X 7 X 7
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                Lambda(lambda x: x.view(-1, 128 * 7 * 7)),
                sn(nn.Linear(128 * 7 * 7, 1024)),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2),
                sn(nn.Linear(1024, 1)),
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class Encoder(nn.Module):
    """Calculates pre-softmax posterior q(c | x)"""

    def __init__(self, n_gen):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(1, 32, 4, 2, 1),                  # B X 32 X 14 X 14
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1),                 # B X 64 X 7 X 7
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                Lambda(lambda x: x.view(-1, 64 * 7 * 7)),
                nn.Linear(64 * 7 * 7, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, n_gen),
        )

        self.n_gen = n_gen

    def forward(self, x):
        return self.main(x)
