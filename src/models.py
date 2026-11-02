import torch
import torch.nn as nn

latent_dim = 100


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # noise -> 512 x 4 x 4
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 512 x 4 x 4 -> 256 x 8 x 8
            UpsampleBlock(512, 256),

            # 256 x 8 x 8 -> 128 x 16 x 16
            UpsampleBlock(256, 128),

            # 128 x 16 x 16 -> 64 x 32 x 32
            UpsampleBlock(128, 64),

            # 64 x 32 x 32 -> 32 x 64 x 64
            UpsampleBlock(64, 32),

            # 32 x 64 x 64 -> 3 x 64 x 64
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    noise = torch.randn(64, latent_dim, 1, 1)
    generator = Generator()
    discriminator = Discriminator()

    fake_images = generator(noise)
    print("Generator output shape:", fake_images.shape)

    predictions = discriminator(fake_images)
    print("Discriminator output shape:", predictions.shape)
                
        
  
    