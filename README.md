# GAN Face Generation with PyTorch

A deep learning project focused on generating synthetic human faces using a Deep Convolutional Generative Adversarial Network (DCGAN), trained on the CelebA dataset.

## Project Overview

This project explores adversarial training for image generation, with a focus on improving training stability and visual quality.

The model consists of two neural networks:
- **Generator:** Produces synthetic face images from random latent vectors.
- **Discriminator:** Distinguishes real images from generated images.

Both networks are trained adversarially.

## Dataset

- **Dataset:** CelebA (CelebFaces Attributes Dataset)
- **Training images:** Approximately 50,000
- **Image resolution:** 64 × 64 pixels
- **Preprocessing:** Resizing, tensor conversion and normalization to [-1, 1]

The dataset is not included in this repository.

## Architecture and Training

The project uses a convolutional GAN architecture implemented in PyTorch.

Training techniques include:

- **Spectral normalization:** Applied to convolutional layers in the discriminator.
- **One-sided label smoothing:** Real labels are set to 0.9.
- **Exponential Moving Average (EMA):** Maintains a smoothed version of generator weights.
- **BCEWithLogitsLoss:** Used for adversarial training.
- **Adam optimization:** With separate learning rates for the generator and discriminator.
- **CUDA acceleration:** Supports training on NVIDIA GPUs.

### Training configuration

| Parameter | Value |
|---|---|
| Latent dimension | 100 |
| Batch size | 64 |
| Generator learning rate | 0.0002 |
| Discriminator learning rate | 0.00005 |
| EMA decay | 0.999 |
| Generator updates per batch | 2 |
| Initial training milestone | 50 epochs |

## Results

At epoch 50, the EMA generator produces recognizable faces with varied appearances, hairstyles and expressions.

Some generated images still exhibit softness and limited fine detail.

Future evaluation will compare generator checkpoints using fixed latent vectors and quantitative image-quality metrics.

## Future Improvements

- Evaluate checkpoints using FID or KID.
- Investigate alternative training configurations.
- Explore higher-resolution image generation.
- Compare standard generator and EMA outputs.

## Technologies

Python, PyTorch, Torchvision, NumPy, Matplotlib and CUDA.

## Repository Structure


gan-face-generation/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── prepare_data.py
│   └── train.py
└── results/
```

## Project Status

Training and evaluation are ongoing. The epoch-50 EMA checkpoint serves as the current baseline.
