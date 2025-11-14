import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neural_blueprints.architectures import CNN
from neural_blueprints.config import CNNConfig, ConvLayerConfig, PoolingLayerConfig, FeedForwardNetworkConfig
from neural_blueprints.utils import Trainer, accuracy


if __name__ == "__main__":
    # Define transformations for MNIST (normalize to [0,1])
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL image to tensor (C x H x W) in [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST normalization
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define CNN configuration
    cnn_config = CNNConfig(
        layer_types=['conv2d', 'pool2d', 'conv2d', 'pool2d', 'flatten'],
        layer_configs=[
            ConvLayerConfig(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            PoolingLayerConfig(pool_type='max', kernel_size=2, stride=2),
            ConvLayerConfig(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            PoolingLayerConfig(pool_type='max', kernel_size=2, stride=2),
            {}  # flatten layer does not need a config
        ],
        feedforward_config=FeedForwardNetworkConfig(input_dim=32 * 7 * 7, hidden_dims=[128], output_dim=10),
        final_activation=None
    )

    # Initialize CNN model
    model = CNN(cnn_config)
    model.blueprint()

    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, optimizer, "cross_entropy")

    # Train the model
    trainer.train(train_loader, test_loader, epochs=2)

    # Evaluate on test set
    # For classification, y_test should be integer labels
    X_test = torch.stack([x for x, _ in test_dataset])
    y_test = torch.tensor([y for _, y in test_dataset])
    test_pred, test_loss = trainer.predict(X_test, y_test)

    test_pred = torch.argmax(test_pred, dim=1)
    assert y_test.shape == test_pred.shape, f"Shape mismatch between true labels and predictions: True = {y_test.shape} vs Pred = {test_pred.shape}"
    test_accuracy = accuracy(test_pred, y_test)
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")