# Neural Blueprints

**Neural Blueprints** is a modular Python package designed to streamline experimentation with deep learning architectures. It provides clean, reusable building blocks for neural networks, along with configuration utilities, preprocessing tools, training helpers, and dataset abstractions.

This project is organized to support rapid prototyping while keeping codebases maintainable and scalable.

---

## 🚀 Features

* **Prebuilt Architectures**: MLPs, CNNs, RNNs, Transformers, GANs, Autoencoders.
* **Composable Components**: Core layers (dense, conv, recurrent, attention, residual, etc.) and higher-level composite modules (encoders, decoders, generators, discriminators).
* **Config System**: Structured configuration objects for architectures and components.
* **Dataset Utilities**: Built-in masked tabular dataset class.
* **Preprocessing Tools**: Tabular preprocessing pipelines.
* **Training Utilities**: Simple trainer abstraction, metrics, device handling, visualization helpers.

---

## 📁 Project Structure

```
neural_blueprints
├── architectures/        # High-level neural network architectures
├── components/           # Core and composite building blocks
├── config/               # Configuration schemas and utilities
├── datasets/             # Dataset classes
├── preprocess/           # Preprocessing utilities
└── utils/                # Training, metrics, device, visualization helpers
```

### Architectures

Located in `architectures/`:

* `mlp.py` — Multi-Layer Perceptron
* `cnn.py` — Convolutional Neural Network
* `rnn.py` — Recurrent Neural Network
* `transformer.py` — Transformer encoder/decoder
* `autoencoder.py` — Autoencoder architecture
* `gan.py` — Generative Adversarial Network (Generator & Discriminator)

### Components

Inside `components/`:

* **core** — atomic layers (attention, conv, dense, embedding, pooling, recurrent, projection, residual, reshape)
* **composite** — higher-level modules (encoder, decoder, generator, discriminator, etc.)

### Config

`config/` defines structured configs for:

* architectures
* composite components
* core components
* shared utilities (e.g., registry systems)

### Datasets

* `masked_tabular_dataset.py` — masked tabular dataset for reconstruction or imputation tasks.

### Preprocess

* `tabular_preprocess.py` — preprocessing tools for tabular datasets.

### Utils

Everything that makes experimentation smooth:

* activation functions
* normalization layers
* blocks & preprocess helpers
* trainer abstraction
* metrics
* device management
* visualization utilities

---

## 📦 Installation

You can install the package locally via:

```bash
pip install -e .
```

If you're using this as a template or integrating it into your own project, simply clone the repository and import modules directly.

---

## 🧪 Usage Example

```python
from neural_blueprints.architectures.mlp import MLP
from neural_blueprints.utils.trainer import Trainer

model = MLP(input_dim=128, hidden_dims=[256, 128, 64], output_dim=10)
trainer = Trainer(model)

trainer.fit(train_loader, epochs=20)
```

---

## 🧱 Philosophy

Neural Blueprints follows three core principles:

1. **Modularity** — Every component is replaceable.
2. **Clarity** — No magic. Explicit is better than implicit.
3. **Reusability** — Build once, compose anywhere.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📄 License

MIT License. Use freely for research, education, and professional work.

---

## ⭐ Acknowledgements

Inspired by modern deep learning libraries and research best practices.