import torch
from ucimlrepo import fetch_ucirepo 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from neural_blueprints.architectures import MLP
from neural_blueprints.config import MLPConfig
from neural_blueprints.utils import Trainer, accuracy, infer_types

if __name__ == "__main__":
    data = fetch_ucirepo(id=2)
    X = data.data.features
    y = data.data.targets

    X_types = infer_types(X)
    y_types = infer_types(y)
    X = X.astype({col: dtype for col, dtype in zip(X.columns, X_types)})
    y = y.astype(y_types[0])

    # Normalize features

    for col in X.select_dtypes(include=['category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    for col in y.select_dtypes(include=['category']).columns:
        le = LabelEncoder()
        y[col] = le.fit_transform(y[col])
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X.to_numpy())

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y.to_numpy().reshape(-1), test_size=0.3, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define model configuration
    mlp_config = MLPConfig(
        input_dim=X_train.shape[1],
        hidden_dims=[64, 32, 16],
        output_dim=y_train.shape[1],
        normalization=None,
        activation='relu',
        final_activation=None
    )

    # Initialize model
    model = MLP(mlp_config)
    model.blueprint()

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(model, optimizer, "mse")

    # Train the model
    trainer.train(train_loader, test_loader, epochs=100)

    # Evaluate on test set
    test_pred, test_loss = trainer.predict(X_test, y_test)
    test_accuracy = accuracy(test_pred, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")