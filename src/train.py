
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from dataset import dataloader
from torchvision.utils import save_image
from pathlib import Path
from models import Generator, Discriminator
from torch.nn.utils.parametrizations import spectral_norm

# Training settings
batch_size = 64
latent_dim = 100
num_epochs = 50
ema_decay = 0.999

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# Create folders for the new architecture
checkpoint_dir = Path("checkpoints_upconv")
sample_dir = Path("Outputs/Samples_upconv")

checkpoint_dir.mkdir(parents=True, exist_ok=True)
sample_dir.mkdir(parents=True, exist_ok=True)

# Fixed noise for comparing generated faces
noise_path = checkpoint_dir / "fixed_noise.pt"

if noise_path.exists():
    fixed_noise = torch.load(noise_path, map_location=device, weights_only=True)
else:
    fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)
    torch.save(fixed_noise.cpu(), noise_path)

# Loss function
loss_func = nn.BCEWithLogitsLoss()

# Initialize Generator and Discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Apply spectral normalization
for layer in discriminator.model:
    if isinstance(layer, nn.Conv2d):
        spectral_norm(layer)

# Load epoch 27 models
generator.load_state_dict(
    torch.load("checkpoints_upconv/generator_epoch_27.pth",
               map_location=device, weights_only=True)
)

discriminator.load_state_dict(
    torch.load("checkpoints_upconv/discriminator_epoch_27.pth",
               map_location=device, weights_only=True)
)

# Continue EMA from epoch 27
ema_generator = copy.deepcopy(generator)

ema_generator.load_state_dict(
    torch.load("checkpoints_upconv/ema_generator_epoch_27.pth",
               map_location=device, weights_only=True)
)

ema_generator.eval()
ema_generator.requires_grad_(False)

# Adam optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))

# Load optimizer states
checkpoint = torch.load(
    "checkpoints_upconv/optimizers_epoch_27.pth",
    map_location=device,
    weights_only=True
)

optimizer_G.load_state_dict(checkpoint["optimizer_G"])
optimizer_D.load_state_dict(checkpoint["optimizer_D"])

# Update EMA weights after Generator training
@torch.no_grad()
def update_ema(ema_generator, generator, decay=0.999):
    for ema_param, param in zip(ema_generator.parameters(), generator.parameters()):
        ema_param.mul_(decay).add_(param.detach(), alpha=1 - decay)

    # Copy BatchNorm buffers
    for ema_buffer, buffer in zip(ema_generator.buffers(), generator.buffers()):
        ema_buffer.copy_(buffer)

# Training loop
for epoch in range(27, num_epochs):
    generator.train()
    discriminator.train()

    epoch_loss_D = 0
    epoch_loss_G = 0

    for real_images, _ in dataloader:
        real_images = real_images.to(device)
        current_batch_size = real_images.size(0)

        real_labels = torch.full((current_batch_size,), 0.9, device=device)
        fake_labels = torch.zeros(current_batch_size, device=device)

        # Train discriminator
        optimizer_D.zero_grad()

        real_pred = discriminator(real_images).view(-1)
        loss_real = loss_func(real_pred, real_labels)

        noise = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)

        fake_pred = discriminator(fake_images.detach()).view(-1)
        loss_fake = loss_func(fake_pred, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # Train generator twice
        discriminator.requires_grad_(False)
        generator_loss_batch = 0

        for g_step in range(2):
            optimizer_G.zero_grad()

            noise = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)

            fake_pred = discriminator(fake_images).view(-1)
            generator_targets = torch.ones_like(fake_pred)

            loss_G = loss_func(fake_pred, generator_targets)
            loss_G.backward()
            optimizer_G.step()

            # Update EMA after each Generator step
            update_ema(ema_generator, generator, ema_decay)
            generator_loss_batch += loss_G.item()

        discriminator.requires_grad_(True)

        # Record losses
        epoch_loss_D += loss_D.item()
        epoch_loss_G += generator_loss_batch / 2

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss D: {loss_D.item():.4f} "
            f"Loss G: {loss_G.item():.4f}"
        )

    # Average losses
    avg_loss_D = epoch_loss_D / len(dataloader)
    avg_loss_G = epoch_loss_G / len(dataloader)

    print(f"Average Loss D: {avg_loss_D:.4f}, Loss G: {avg_loss_G:.4f}")

    # generate sample images
    generator.eval()
    ema_generator.eval()

    with torch.no_grad():
        normal_samples = generator(fixed_noise)
        ema_samples = ema_generator(fixed_noise)

        save_image(
            normal_samples,
            sample_dir / f"epoch_{epoch + 1}.png",
            nrow=4, normalize=True, value_range=(-1, 1)
        )

        save_image(
            ema_samples,
            sample_dir / f"ema_epoch_{epoch + 1}.png",
            nrow=4, normalize=True, value_range=(-1, 1)
        )

    # Save model checkpoints
    torch.save(generator.state_dict(), checkpoint_dir / f"generator_epoch_{epoch + 1}.pth")
    torch.save(discriminator.state_dict(), checkpoint_dir / f"discriminator_epoch_{epoch + 1}.pth")
    torch.save(ema_generator.state_dict(), checkpoint_dir / f"ema_generator_epoch_{epoch + 1}.pth")

    # Save optimizer states
    torch.save({
        "epoch": epoch + 1,
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict()
    }, checkpoint_dir / f"optimizers_epoch_{epoch + 1}.pth")

    print(f"Epoch {epoch + 1}/{num_epochs} completed.")

print("Training completed.")

