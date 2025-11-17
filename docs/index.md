# Neural Blueprints

**Neural Blueprints** is a modular Python package designed to streamline experimentation with deep learning architectures. It provides clean, reusable building blocks for neural networks, along with configuration utilities, preprocessing tools, training helpers, and dataset abstractions.

## 🚀 Key Features

* **Prebuilt Architectures**: MLPs, CNNs, RNNs, Transformers, GANs, Autoencoders
* **Composable Components**: Core layers and higher-level composite modules
* **Config System**: Structured configuration objects for reproducible experiments
* **Dataset Utilities**: Built-in dataset classes for common tasks
* **Preprocessing Tools**: Tabular preprocessing pipelines
* **Training Utilities**: Simple trainer abstraction, metrics, and visualization

## 🎯 Philosophy

Neural Blueprints follows three core principles:

1. **Modularity** — Every component is replaceable
2. **Clarity** — No magic. Explicit is better than implicit
3. **Reusability** — Build once, compose anywhere

## 📦 Installation
```bash
pip install neural-blueprints
```

Or install from source:
```bash
git clone https://github.com/elnukakujo/neural-blueprints.git
cd neural-blueprints
pip install -e .
```

## 🚦 Quick Start
```python
from neural_blueprints.architectures import MLP
from neural_blueprints.config import MLPConfig, TrainerConfig
from neural_blueprints.utils import Trainer

# Define model configuration
mlp_config = MLPConfig(
    input_dim=X.shape[1],
    hidden_dims=[64, 32, 16],
    output_dim=1,
    normalization=None,
    activation='relu',
    final_activation=None
)

# Initialize model
model = MLP(mlp_config)
model.blueprint()

trainer = Trainer(
    config=TrainerConfig(
        training_type="label",
        optimizer="adam",
        criterion="mse",
        learning_rate=0.001,
        weight_decay=1e-5,
        batch_size=32,
        save_weights_path="../models/mlp_adult.pth"
    ),
    model= model
)

# Train the model
trainer.train(train_dataset, val_dataset, epochs=5)
```

## 📚 Documentation Structure

- **[Getting Started](getting-started.md)** - Installation and basic usage
- **[User Guide](user-guide/architectures.md)** - Detailed guides for each module
- **[API Reference](reference/neural_blueprints/)** - Complete API documentation

## 🏗️ Project Structure

```
neural_blueprints
├── architectures/        # High-level neural network architectures
├── components/           # Core and composite building blocks
├── config/               # Configuration schemas and utilities
├── datasets/             # Dataset classes
├── preprocess/           # Preprocessing utilities
└── utils/                # Training, metrics, device, visualization helpers
```

## 📄 License

MIT License. Use freely for research, education, and professional work.

## 🌟 Acknowledgements

Inspired by modern deep learning libraries and research best practices.