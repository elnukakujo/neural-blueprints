"""Example: Training a Variational AutoEncoder"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from neural_blueprints.feedforward import VariationalAE


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Create VAE
    vae = VariationalAE(
        input_dim=784,
        hidden_dims=[512, 256],
        latent_dim=20,
        activation="relu"
    )
    vae.to(device)
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = vae(data)
            loss = vae.compute_loss(data, recon, mu, logvar, beta=1.0)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Generate samples
        if (epoch + 1) % 5 == 0:
            vae.eval()
            with torch.no_grad():
                samples = vae.sample(64)
                samples = samples.view(64, 1, 28, 28)
                save_image(samples, f'vae_samples_epoch_{epoch+1}.png', nrow=8)
    
    # Save model
    vae.save("vae_mnist.pth")
    print("Training complete!")


if __name__ == "__main__":
    main()