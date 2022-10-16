import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from modules import UNet

from utils import getData, saveImages, setupLogging

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class DiffusionModel:
    def __init__(self, noise_steps = 1000, beta_start=1e-4, beta_end=0.02, img_size = 64, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepareNoiseSchedule().to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def prepareNoiseSchedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noiseImage(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat* e, e
    
    def sampleTimesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, model, n):
        logging.info(f'Sampling {n} new images.....')
        model.eval()
        with torch.no_grad():
            x = torch.randn(n, 3, self.img_size, self.img_size).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1-alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta)
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
         



def train(args):
    setupLogging(args.run_name)
    device = args.device
    dataloader = getData(args)
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    criterion = nn.MSELoss()
    diffusion = DiffusionModel(img_size= args.img_size, device= device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for ep in range(args.epochs):
        logging.info(f'Epoch {ep}:')
        pbar = tqdm(dataloader)
        for i, (img, _) in enumerate(pbar):
            img = img.to(device)
            t = diffusion.sampleTimesteps(img.shape[0]).to(device)
            x_t, noise = diffusion.noiseImage(img, t)
            pred_noise = model(x_t, t)
            loss = criterion(noise, pred_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=ep*l + i)
        
        sampled_img = diffusion.sample(model, n=img.shape[0])
        saveImages(sampled_img, os.path.join("Results", args.run_name, f'{ep}.jpg'))
        torch.save(model.state_dict(). os.path.join("models", args.run_name, f'ckpt.pt'))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM Unconditional"
    args.epochs = 500
    args.batch_size = 12
    args.img_size = 64
    args.dataset_path = ""
    args.device = "cuda"
    args.lr = 3e-4
    train(args)

if __name__ == "__main__":
    launch()