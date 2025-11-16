import torch
from sklearn.datasets import fetch_openml

from neural_blueprints.utils import Trainer, accuracy
from neural_blueprints.config import BERTConfig, TransformerEncoderConfig, NormalizationConfig, TrainerConfig
from neural_blueprints.architectures import BERT
from neural_blueprints.datasets import MaskedTabularDataset
from neural_blueprints.preprocess import TabularPreprocessor

# Fetch Adult Income dataset from OpenML
adult = fetch_openml("adult", version=2, as_frame=True)

# X = features, y = target
X = adult.data
y = adult.target

data = X.copy()
data[y.name] = y

# ----------------------------
# 1. Data Preprocessing
# ----------------------------

preprocessor = TabularPreprocessor()
data, discrete_features, continuous_features = preprocessor.run(data)

# Create dataset
dataset = MaskedTabularDataset(
    data, 
    discrete_features, 
    continuous_features,
    mask_prob=0.25
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# ----------------------------
# 2. BERT Configuration
# ----------------------------

bert_config = BERTConfig(
    cardinalities=dataset.cardinalities,
    encoder_config=TransformerEncoderConfig(
        input_dim=data.shape[1],
        hidden_dim=32,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        projection=None,
        final_normalization=NormalizationConfig(norm_type="layernorm", num_features=32),
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

# ----------------------------
# 3. Training
# ---------------------------

trainer = Trainer(
    model=model,
    config=TrainerConfig(
        training_type='masked_label',
        criterion='tabular_masked_loss',
        optimizer='adam',
        learning_rate=1e-3,
        weight_decay=1e-5,
        save_weights_path="models/bert_adult.pth",
        batch_size=64
    )
)
trainer.train(train_dataset, val_dataset, epochs=2, visualize=True)

# ----------------------------
# 4. Evaluation
# ---------------------------

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)
X, y, mask = next(iter(val_loader))  # Get first 100 samples from validation set
predictions, _ = trainer.predict(X, y, mask)

dis_accuracy = 0
cont_accuracy = 0
for column_idx, column_name in enumerate(data.columns):
    print(f"\nFeature Column {column_name}:")
    predicted_attributes = predictions[column_idx]      # shape: (batch_size, num_classes)
    targets = y[:, column_idx]                     # shape: (batch_size,)

    feature_mask = mask[:, column_idx]                  # shape: (batch_size,)
    predicted_attributes = predicted_attributes[feature_mask]
    if predicted_attributes.size(1) > 1:
        predicted_attributes = predicted_attributes.softmax(dim=-1).argmax(dim=-1).cpu().numpy()
    else:
        predicted_attributes = predicted_attributes.squeeze(-1).cpu().numpy()
    targets = targets[feature_mask].cpu().numpy()

    print("Predicted attribute values:", predicted_attributes[:5])
    print("True attribute values:", targets[:5])

    accuracy_value = accuracy(torch.tensor(predicted_attributes), torch.tensor(targets))
    print(f"Accuracy: {accuracy_value:.4f}")
    if column_name in discrete_features:
        dis_accuracy += accuracy_value
    else:
        cont_accuracy += accuracy_value

avg_dis_accuracy = dis_accuracy / len(discrete_features) if len(discrete_features) > 0 else 0
avg_cont_accuracy = cont_accuracy / len(continuous_features) if len(continuous_features) > 0 else 0
print(f"\nAverage Discrete Accuracy: {avg_dis_accuracy:.4f}")
print(f"Average Continuous Accuracy: {avg_cont_accuracy:.4f}")
avg_accuracy = (dis_accuracy + cont_accuracy) / len(data.columns)
print(f"Overall Average Accuracy: {avg_accuracy:.4f}")