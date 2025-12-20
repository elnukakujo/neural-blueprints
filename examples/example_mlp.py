import torch
from ucimlrepo import fetch_ucirepo 
from torch.utils.data import TensorDataset

from neural_blueprints.architectures import MLP
from neural_blueprints.config import MLPConfig, TrainerConfig
from neural_blueprints.utils import Trainer, accuracy, infer_types
from neural_blueprints.preprocess import TabularPreprocessor

data = fetch_ucirepo(id=2)
X = data.data.features
y = data.data.targets

data = X.copy()
data['income'] = y

dtypes = infer_types(data)
data = data.astype(dtypes)

preprocessor = TabularPreprocessor()
data, discrete_features, continuous_features = preprocessor.run(data)
data = data[0]

X = data.drop(columns=['income'])
y = data['income'].values

dataset = TensorDataset(torch.tensor(X.values, dtype=torch.float32), torch.tensor(y.reshape(-1, 1), dtype=torch.float32))

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

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