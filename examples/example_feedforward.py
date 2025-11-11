"""Example: Training a FeedForward Network on MNIST"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neural_blueprints.feedforward import FeedForwardNN
from neural_blueprints.utils import Trainer


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Create model
    model = FeedForwardNN(
        input_dim=784,
        hidden_dims=[512, 256, 128],
        output_dim=10,
        activation="relu",
        dropout=0.2,
        batch_norm=True
    )
    
    print(f"Model has {model.count_parameters():,} parameters")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(model, optimizer, criterion, device)
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=10
    )
    
    # Test
    test_metrics = trainer.evaluate(test_loader)
    print(f"\nFinal Test Accuracy: {test_metrics['accuracy']:.2f}%")
    
    # Save model
    model.save("mnist_ffn.pth", metadata={"accuracy": test_metrics['accuracy']})


if __name__ == "__main__":
    main()