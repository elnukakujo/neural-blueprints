import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import fetch_openml

from neural_blueprints.utils import Trainer, get_device, infer_types
from neural_blueprints.config import BERTConfig, TransformerEncoderConfig, FeedForwardNetworkConfig, NormalizationConfig
from neural_blueprints.architectures import BERT

# Fetch Adult Income dataset from OpenML
adult = fetch_openml("adult", version=2, as_frame=True)

# X = features, y = target
X = adult.data
y = adult.target

data = X.copy()
data[y.name] = y

# Infer types
data_types = infer_types(data)
data_dtypes = {col: dtype for col, dtype in zip(data.columns, data_types)}
data = data.astype(data_dtypes)

# Separate categorical and continuous columns
cat_cols = data.select_dtypes(include=['category']).columns.tolist()
cont_cols = data.select_dtypes(include=['float32', 'int32']).columns.tolist()

# Encode categorical columns
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Standardize continuous columns
scaler = StandardScaler()
data[cont_cols] = scaler.fit_transform(data[cont_cols])

# ----------------------------
# 2️⃣ Masked Attribute Dataset
# ----------------------------
class MaskedTabularDataset(Dataset):
    def __init__(self, data, cat_cols, cont_cols, mask_prob=0.15):
        self.data = torch.tensor(data.to_numpy())
        self.cardinalities = [data[col].nunique() if data[col].dtype.name == 'category' else 1 for col in data.columns]
        self.cat_idx = [data.columns.get_loc(c) for c in cat_cols]
        self.cont_idx = [data.columns.get_loc(c) for c in cont_cols]
        self.mask_prob = mask_prob
        self.is_cat = torch.BoolTensor([True if i in self.cat_idx else False for i in range(data.shape[1])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx].clone()
        mask = torch.rand(row.shape) < self.mask_prob
        labels = row.clone()
        row[mask] = -1  # Masked features
        labels[~mask] = -1  # Only predict masked
        return row, labels, mask

dataset = MaskedTabularDataset(data, cat_cols, cont_cols, mask_prob=0.15)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

# ----------------------------
# 2. BERT Configuration
# ----------------------------

hidden_dim = 32
seq_len = data.shape[1]

bert_config = BERTConfig(
    seq_len=seq_len,
    cardinalities=dataset.cardinalities,
    encoder_config=TransformerEncoderConfig(
        input_dim=seq_len,
        hidden_dim=hidden_dim,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        projection=None,
        final_normalization=NormalizationConfig(norm_type="layernorm", num_features=hidden_dim),
        final_activation=None
    ),
    with_input_projection=True,
    with_output_projection=True,
    dropout=0.1,
    final_normalization=None,
    final_activation=None
)

model = BERT(bert_config)
model.blueprint()

trainer = Trainer(
    model=model,
    optimizer=optim.Adam(model.parameters(), lr=1e-4),
    criterion="cross_entropy",
    device=get_device(),
    is_masked_label=True
)
trainer.train(train_loader, val_loader, epochs=5, visualize=False)

# Example prediction
model.eval()
with torch.no_grad():
    tokens, labels, mask = next(iter(val_loader))
    tokens = tokens.to(get_device())
    predictions = model(tokens)

    for column_idx in range(len(predictions)):
        predicted_attributes = predictions[column_idx]      # shape: (batch_size, num_classes)
        targets = labels[:, column_idx]                     # shape: (batch_size,)

        feature_mask = mask[:, column_idx]                  # shape: (batch_size,)
        predicted_attributes = predicted_attributes[feature_mask]
        targets = targets[feature_mask]
        print("Predicted token IDs at masked positions:", predicted_attributes.argmax(dim=1).cpu().numpy())