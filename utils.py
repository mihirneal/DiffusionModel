import os
import torch
import torchvision
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def plotImages(img):
    plt.figure(figsize=(32,32))
    plt.imshow(torch.cat([
        torch.cat([i for i in img.cpu()], dim=-1)
    ], dim = -2).permute(1, 2, 0).cpu())
    plt.show()

def saveImages(img, path, **kwargs):
    grid = torchvision.utils.make_grid(img, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu")
    im = Image.fromarray(ndarr)
    im.save(path)

def getData(args):
    img_transforms = transforms.Compose([
        transforms.Resize(80),
        transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=img_transforms)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True)
    return dataloader


def setupLogging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)