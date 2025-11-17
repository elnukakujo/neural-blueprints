# Configuration System
This module defines a unified configuration system used across all model components, layers, and training utilities.

Each configuration class is a **Pydantic schema** that formalizes the structure, constraints, and hyperparameters required to build a model or subsystem.

Configurations follow a consistent pattern:

- They describe what the model or layer should look like
- They enforce validation rules
- They are consumed by constructors in `architectures.py`, `components/composite.py`, `components/core.py`, and `utils/`
- They are fully documented in the [API Reference](../../reference/neural_blueprints/) section

---

## 📁 Module Structure
The configuration schemas are organized into three categories: core layers, composite modules, and full architectures, with optional training utilities.

```bash
config/
├── __init__.py
├── core.py               # Primitive layer configs
├── composite.py          # Multi-layer or block configs
├── architectures.py      # High-level model definitions
└── utils.py              # Training and runtime configs
```

---

### 1. Core Layer Configurations
These represent the smallest configurable building blocks.

They typically correspond to operations used inside larger modules.

Included classes:

- [`DenseLayerConfig`](../../reference/neural_blueprints/config/core/#neural_blueprints.config.core.DenseLayerConfig)
- [`ConvLayerConfig`](../../reference/neural_blueprints/config/core/#neural_blueprints.config.core.ConvLayerConfig)
- [`RecurrentUnitConfig`](../../reference/neural_blueprints/config/core/#neural_blueprints.config.core.RecurrentUnitConfig)
- [`AttentionLayerConfig`](../../reference/neural_blueprints/config/core/#neural_blueprints.config.core.AttentionLayerConfig)
- [`ResidualLayerConfig`](../../reference/neural_blueprints/config/core/#neural_blueprints.config.core.ResidualLayerConfig)
- [`EmbeddingLayerConfig`](../../reference/neural_blueprints/config/core/#neural_blueprints.config.core.EmbeddingLayerConfig)
- [`PatchEmbeddingLayerConfig`](../../reference/neural_blueprints/config/core/#neural_blueprints.config.core.PatchEmbeddingLayerConfig)
- [`PoolingLayerConfig`](../../reference/neural_blueprints/config/core/#neural_blueprints.config.core.PoolingLayerConfig)
- [`NormalizationConfig`](../../reference/neural_blueprints/config/core/#neural_blueprints.config.core.NormalizationLayerConfig)
- [`ReshapeLayerConfig`](../../reference/neural_blueprints/config/core/#neural_blueprints.config.core.ReshapeLayerConfig)
- [`ProjectionLayerConfig`](../../reference/neural_blueprints/config/core/#neural_blueprints.config.core.ProjectionLayerConfig)

These configs specify essential parameters like dimensionality, activation functions, kernel sizes, normalization choices, and other architectural details.

They are typically not used directly by end users but serve as the foundation for higher-level configurations.

---

### 2. Composite Module Configurations
These describe multi-layer or structured components that combine several core layers.

Available composites:

- [`FeedForwardNetworkConfig`](../../reference/neural_blueprints/config/composite/#neural_blueprints.config.composite.FeedForwardNetworkConfig)
- [`EncoderConfig`](../../reference/neural_blueprints/config/composite/#neural_blueprints.config.composite.EncoderConfig)
- [`DecoderConfig`](../../reference/neural_blueprints/config/composite/#neural_blueprints.config.composite.DecoderConfig)
- [`GeneratorConfig`](../../reference/neural_blueprints/config/composite/#neural_blueprints.config.composite.GeneratorConfig)
- [`DiscriminatorConfig`](../../reference/neural_blueprints/config/composite/#neural_blueprints.config.composite.DiscriminatorConfig)
- [`TransformerEncoderConfig`](../../reference/neural_blueprints/config/composite/#neural_blueprints.config.composite.TransformerEncoderConfig)
- [`TransformerDecoderConfig`](../../reference/neural_blueprints/config/composite/#neural_blueprints.config.composite.TransformerDecoderConfig)

These configs assemble internal submodules, such as stacked dense layers, attention blocks, or transformer stages.

They often accept lists of core configs or high-level parameters that get expanded into internal structures.

---

### 3. Architecture-Level Configurations
These define complete models.

They aggregate multiple composite configs and govern the data flow across the full architecture.

Architecture configs include:

- [`MLPConfig`](../../reference/neural_blueprints/config/architectures/#neural_blueprints.config.architectures.MLPConfig)
- [`CNNConfig`](../../reference/neural_blueprints/config/architectures/#neural_blueprints.config.architectures.CNNConfig)
- [`RNNConfig`](../../reference/neural_blueprints/config/architectures/#neural_blueprints.config.architectures.RNNConfig)
- [`AutoEncoderConfig`](../../reference/neural_blueprints/config/architectures/#neural_blueprints.config.architectures.AutoEncoderConfig)
- [`GANConfig`](../../reference/neural_blueprints/config/architectures/#neural_blueprints.config.architectures.GANConfig)
- [`TransformerConfig`](../../reference/neural_blueprints/config/architectures/#neural_blueprints.config.architectures.TransformerConfig)
- [`TabularBERTConfig`](../../reference/neural_blueprints/config/architectures/#neural_blueprints.config.architectures.TabularBERTConfig)

These schemas describe the full topology of a model—number of layers, embedding strategies, architectural variants, recurrent units, attention stacks, or any other structural options.

Each architecture config maps cleanly onto a corresponding instantiation in `architectures/`

---

### 4. Training and Utility Configurations

- [`TrainerConfig`](../../reference/neural_blueprints/config/utils/#neural_blueprints.config.utils.TrainerConfig)

This configuration defines training-related hyperparameters such as optimization, scheduling, evaluation intervals, checkpointing, and general runtime behavior.

---

## 🎯 Design Principles
The configuration system is built around the following principles:

### Declarative Model Specification
All model structure is declared through configuration objects rather than imperative code.

This ensures transparency, reproducibility, and clean separation between model definition and implementation.

### Validation and Safety

Configs validate inputs—dimensions must match, options must be supported, and layer definitions must be coherent.

### Modularity
Higher-level configs compose lower-level ones.

Users can configure an entire transformer or GAN with a single object, while advanced users can provide fine-grained nested configs.

### Interoperability
All config classes share the same Pydantic foundation and follow similar naming conventions, making them predictable and easy to integrate.

---

## 📘 Usage Example (Abstract)
```python
from neural_blueprints.config import MLPConfig, TrainerConfig
from neural_blueprints.utils import Trainer
from neural_blueprints.architectures import MLP

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
    config = TrainerConfig(
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

trainer.train(train_dataset, val_dataset, epochs = 5, visualize=False)
```

---

## 📚 API Reference
Full parameter documentation, default values, validation details, and examples for every configuration class are available in the [API Reference](../../reference/neural_blueprints/) section.

Use this page as a conceptual map; refer to the API Reference for class-specific details.