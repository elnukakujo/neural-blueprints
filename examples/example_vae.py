import torch
from torchvision import datasets, transforms

from neural_blueprints.architectures.autoencoder import VariationalAutoEncoder
from neural_blueprints.config import AutoEncoderConfig, ConvLayerConfig, DenseLayerConfig, ReshapeLayerConfig, TrainerConfig
from neural_blueprints.utils import Trainer, get_device, image_plot

# ----------------------------
# 1. Data preparation
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

latent_dim = 20

vae_config = AutoEncoderConfig(
    encoder_layer_types=['conv2d', 'conv2d', 'flatten', 'dense', 'dense'],
    encoder_layer_configs=[
        ConvLayerConfig(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),    # 28x28 → 14x14
        ConvLayerConfig(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),   # 14x14 → 7x7
        {},                                                                                     # flatten
        DenseLayerConfig(input_dim=64*7*7, output_dim=512),                                     # 64 * 7 * 7 → 512
        DenseLayerConfig(input_dim=512, output_dim=latent_dim*2)                                # 512 → latent_dim * 2 (mean + logvar)
    ],
    decoder_layer_types=['dense', 'dense', 'reshape', 'conv2d_transpose', 'conv2d_transpose'],
    decoder_layer_configs=[
        DenseLayerConfig(input_dim=latent_dim, output_dim=512),                                                     # latent → 512
        DenseLayerConfig(input_dim=512, output_dim=64*7*7),                                                         # 512 → 64 * 7 * 7
        ReshapeLayerConfig(shape=(64,7,7)),                                                                         # reshape to (64,7,7)
        ConvLayerConfig(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),     # 7x7→14x14
        ConvLayerConfig(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1, activation='sigmoid')  # 14x14→28x28
    ]
)

model = VariationalAutoEncoder(vae_config)
model.blueprint()

trainer = Trainer(
    config=TrainerConfig(
        learning_rate=1e-3,
        weight_decay=1e-5,
        save_weights_path="../models/vae_mnist.pth",
        batch_size=64,
        training_type="reconstruction",
        criterion="vae_loss",
        optimizer='adam'
    ),
    model=model
)
trainer.train(train_dataset=train_dataset, val_dataset=test_dataset, epochs=10)

model.eval()
with torch.no_grad():
    samples = model.decode(torch.randn(16, latent_dim).to(get_device())).cpu()
    samples = samples.permute(0, 2, 3, 1)  # Convert from (N, C, H, W) to (N, H, W, C)

image_plot(images=samples, title="Generated MNIST Digits", cols=4)