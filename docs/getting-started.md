# Getting Started

This guide will help you get started with Neural Blueprints.

## Installation

### Using pip
```bash
pip install neural-blueprints
```

### From Source
```bash
git clone https://github.com/elnukakujo/neural-blueprints.git
cd src/neural-blueprints
pip install -e .
```

### Requirements

- Python 3.12+
- PyTorch 2.9+
- Plotly 6.4+
- Additional dependencies are automatically installed

## Basic Concepts

### Configurations
Neural Blueprints uses strict configuration classes (based on Pydantic) to guarantee correctness before any component or model is built. Configurations:

- Validate user input
- Become immutable typed objects
- Encode the “blueprint” for all components
- Enable clear, testable invariants
- Simplify debugging and improving observability

Example: 
```python
from typing import List, Optional
from pydantic import BaseModel, model_validator

class EncoderConfig(BaseModel):
    """Configuration for an encoder composed of multiple layers."""

    layer_types: List[str]
    layer_configs: List[BaseModel]
    projection: Optional[ProjectionLayerConfig] = None
    final_activation: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self):
        if len(self.layer_types) != len(self.layer_configs):
            raise ValueError("layer_types and layer_configs must have the same length")
        supported_layers = {'dense', 'conv1d', 'conv2d', 'recurrent', 'attention', 'flatten', 'reshape'}
        for layer_type in self.layer_types:
            if layer_type.lower() not in supported_layers:
                raise ValueError(f"Unsupported layer type: {layer_type}. Supported types: {supported_layers}")
        if self.projection is not None:
            if self.projection.input_dim <= 0 or self.projection.output_dim <= 0:
                raise ValueError("projection input_dim and output_dim must be positive integers if specified")
        if self.final_activation is not None and self.final_activation.lower() not in ('relu', 'tanh', 'sigmoid', 'softmax', 'gelu'):
            raise ValueError(f"Unsupported final_activation: {self.final_activation}. Supported: {'relu', 'tanh', 'sigmoid', 'softmax', 'gelu'}")
        return self
```

➡️ Learn more about: [Configuration](user-guide/configuration.md)

### Components

Components are the **building blocks** used to create neural networks.
There are two categories: **Core Components** and **Composite Components**.

#### Core Components

These are the low-level modules: **Dense**, **Conv1d/2d** (+**Transpose**), **Embeddings**, **Attention**, **Reshape**, etc.

```python
from neural_blueprints.components.core import DenseLayer, Conv2dLayer
from neural_blueprints.config import DenseLayerConfig, ConvLayerConfig

dense = DenseLayer(
    config = DenseLayerConfig(
        input_dim = 128,
        output_dim = 64,
        normalization = None,
        activation = 'relu'
    )
)
conv = Conv2dLayer(
    config = ConvLayerConfig(
        in_channels = 3,
        out_channels = 64,
        kernel_size = 3
    )
)
```
➡️ Learn more about: [Core Components](user-guide/components.md)

#### Composite Components

Composite components are larger modules created by combining core components.
Example: the built-in `Encoder`.

```python
from neural_blueprints.components.composite import Encoder
from neural_blueprints.config import EncoderConfig, ConvLayerConfig, DenseLayerConfig

encoder_config = EncoderConfig(
    layer_types=['conv2d', 'conv2d', 'flatten', 'dense', 'dense'],
    layer_configs=[
        ConvLayerConfig(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
        ConvLayerConfig(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
        {},
        DenseLayerConfig(input_dim=64*7*7, output_dim=512),
        DenseLayerConfig(input_dim=512, output_dim=40)
    ]
    projection = None
    final_activation = None
)

encoder = Encoder(config = encoder_config)
```
➡️ Learn more about: [Composite Components](user-guide/components.md)

### Architectures

Architectures are full end-to-end networks built from components.
Example: Variational Autoencoder (VAE).

```python
from neural_blueprints.architectures.autoencoder import VariationalAutoEncoder
from neural_blueprints.config import VariationalAutoEncoderConfig, ConvLayerConfig, DenseLayerConfig, ReshapeLayerConfig

latent_dim = 20

vae_config = VariationalAutoEncoderConfig(
    encoder_layer_types=['conv2d', 'conv2d', 'flatten', 'dense', 'dense'],
    encoder_layer_configs=[
        ConvLayerConfig(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
        ConvLayerConfig(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
        {},
        DenseLayerConfig(input_dim=64*7*7, output_dim=512),
        DenseLayerConfig(input_dim=512, output_dim=latent_dim*2)
    ],
    decoder_layer_types=['dense', 'dense', 'reshape', 'conv2d_transpose', 'conv2d_transpose'],
    decoder_layer_configs=[
        DenseLayerConfig(input_dim=latent_dim, output_dim=512),
        DenseLayerConfig(input_dim=512, output_dim=64*7*7),
        ReshapeLayerConfig(shape=(64,7,7)),
        ConvLayerConfig(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
        ConvLayerConfig(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1, activation='sigmoid')
    ]
)

model = VariationalAutoEncoder(
    config = vae_config
)
model.blueprint()
```
➡️ Learn more about: [Architectures](user-guide/architectures.md)

### Training your Model

Training is handled by the Trainer, which uses a validated TrainerConfig.

#### Trainer Configuration

```python
class TrainerConfig(BaseModel):
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    save_weights_path: Optional[str] = None
    training_type: str = "label"  # Options: 'reconstruction', 'masked_label', 'label'
    criterion: str = "mse"  # Options: 'mse', 'mae', 'cross_entropy', etc.
    optimizer: str = "adam"  # Options: 'adam', 'sgd', etc.

    @model_validator(mode="after")
    def _validate(self):
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("Weight decay cannot be negative.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        valid_types = ['reconstruction', 'masked_label', 'label']
        if self.training_type not in valid_types:
            raise ValueError(f"Invalid training_type: {self.training_type}. Must be one of {valid_types}.")
        if self.training_type == 'reconstruction' and self.criterion not in ['mse', 'mae', 'vae_loss']:
            raise ValueError("For 'reconstruction' training_type, criterion must be one of ['mse', 'mae', 'vae_loss'].")
        if self.training_type == 'masked_label' and self.criterion not in ['tabular_masked_loss']:
            raise ValueError("For 'masked_label' training_type, criterion must be 'tabular_masked_loss'.")
        if self.training_type == 'label' and self.criterion not in ['cross_entropy', 'binary_cross_entropy', 'mse', 'mae', 'rmse']:
            raise ValueError("For 'label' training_type, criterion must be one of ['cross_entropy', 'binary_cross_entropy', 'mse', 'mae', 'rmse'].")
        return self
```

#### Training Modes

| Training Type      | Use Case                          | Allowed Losses                     |
| ------------------ | --------------------------------- | ---------------------------------- |
| **reconstruction** | Autoencoders, VAEs, denoisers     | mse, mae, vae_loss                 |
| **masked_label**   | Masked modeling for tabular tasks | tabular_masked_loss                |
| **label**          | Classification & Regression       | cross_entropy, bce, mse, mae, rmse |

#### Example Training Loop

```python
from neural_blueprints.utils import Trainer
from neural_blueprints.config import TrainerConfig

trainer = Trainer(
    model=model,
    config=TrainerConfig(
        learning_rate=1e-3,
        batch_size=64,
        training_type='reconstruction',
        criterion='mse',
        save_weights_path='./models/vae_weights.pth'
    )
)

trainer.train(train_dataset, val_dataset, epochs = 5, visualize=True)

```

## Next Steps

- Explore [Architectures](user-guide/architectures.md) to learn about available models
- Learn about [Components](user-guide/components.md) for custom architectures
- Check out [Examples](examples/mlp-classification.md) for more use cases
- Read the [API Reference](../reference/neural_blueprints) for detailed documentation