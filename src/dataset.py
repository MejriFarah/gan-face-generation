import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

transform=transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
dataset=datasets.ImageFolder(root="./data",transform=transform)
print(f"We have {len(dataset)} images.")
dataloader=DataLoader(dataset=dataset,batch_size=64,shuffle=True)
if __name__ == "__main__":
 images,_=next(iter(dataloader))
 print(f"Batch shape :",{images.shape})
 grid=make_grid(images[:16],nrow=4,normalize=True)
 plt.figure(figsize=(12,12))
 plt.imshow(grid.permute(1,2,0))
 plt.axis("off")
 os.makedirs("Outputs/Samples",exist_ok=True)
 plt.savefig("Outputs/Samples/Real_faces.png")
 #plt.show()