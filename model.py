import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.dim_z = config.latent_dim
        self.ch = config.g_ch
        
        self.linear = nn.Linear(self.dim_z, 4 * 4 * 8 * self.ch)
        
        self.main = nn.Sequential(
            # 4x4
            nn.BatchNorm2d(8 * self.ch),
            nn.ReLU(True),
            nn.ConvTranspose2d(8 * self.ch, 4 * self.ch, 4, 2, 1),
            # 8x8
            nn.BatchNorm2d(4 * self.ch),
            nn.ReLU(True),
            nn.ConvTranspose2d(4 * self.ch, 2 * self.ch, 4, 2, 1),
            # 16x16
            nn.BatchNorm2d(2 * self.ch),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * self.ch, self.ch, 4, 2, 1),
            # 32x32
            nn.BatchNorm2d(self.ch),
            nn.ReLU(True),
            nn.Conv2d(self.ch, 3, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, z):
        x = self.linear(z)
        x = x.view(-1, 8 * self.ch, 4, 4)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.ch = config.d_ch
        
        self.main = nn.Sequential(
            # 32x32
            nn.Conv2d(3, self.ch, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ch, self.ch, 4, 2, 1),
            # 16x16
            nn.BatchNorm2d(self.ch),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ch, 2 * self.ch, 3, 1, 1),
            nn.BatchNorm2d(2 * self.ch),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * self.ch, 2 * self.ch, 4, 2, 1),
            # 8x8
            nn.BatchNorm2d(2 * self.ch),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * self.ch, 4 * self.ch, 3, 1, 1),
            nn.BatchNorm2d(4 * self.ch),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(4 * self.ch, 4 * self.ch, 4, 2, 1),
            # 4x4
            nn.BatchNorm2d(4 * self.ch),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(4 * self.ch, 1, 4, 1, 0),
        )
        
    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1) 