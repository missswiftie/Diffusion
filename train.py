import torch
import torch.nn as nn
import torch.nn.functional as F
from UNet import UNet
from difussion import DiffusionModel
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

CHANNEL=1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 64
learning_rate = 1e-5
num_epochs = 10
n_steps = 1000
image_size = 32
n_channels = 64

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

unet = UNet(
    image_channels=CHANNEL,
    n_channels=n_channels,
    channel_times=[1, 2, 2, 4],
    use_attn=[False, False, True, True],
    n_blocks=2
).to(device)

diffusion_model = DiffusionModel(
    model=unet,
    n_steps=n_steps,
    beta_start=1e-4,
    beta_end=0.02,
    device=device
).to(device)

optimizer = optim.Adam(diffusion_model.parameters(), lr=learning_rate)

os.makedirs("results", exist_ok=True)
os.makedirs("samples", exist_ok=True)

def train():
    diffusion_model.train()
    total_loss=0.0
    for i,(images,_) in enumerate(tqdm(dataloader)):
        images=images.to(device)
        optimizer.zero_grad()
        loss=diffusion_model(images)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    return total_loss/len(dataloader)

@torch.no_grad()
def sample(num_samples=16):
    diffusion_model.eval()
    samples = diffusion_model.sample(num_samples, image_size, image_size, CHANNEL)
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    grid = make_grid(samples, nrow=4)
    return grid

def save_images(grid, epoch):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.savefig(f'samples/samples_epoch_{epoch:03d}.png')
    plt.close()

if __name__=="__main__":
        

    best_loss = float('inf')
    losses = []

    for epoch in range(1, num_epochs + 1):
        avg_loss = train()
        losses.append(avg_loss)
        
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        if epoch % 1 == 0 or epoch == 1:
            sample_grid = sample(16)
            save_images(sample_grid, epoch)
        
    best_loss = avg_loss
    torch.save({
        'epoch': epoch,
        'model_state_dict': diffusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss,
    }, 'results/best_model.pth')

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('results/loss_curve.png')
    plt.close()
    print("Training completed!")