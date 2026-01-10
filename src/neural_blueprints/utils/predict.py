import torch
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression

from .metrics import accuracy
from ..architectures.base import EncoderArchitecture

import logging
logger = logging.getLogger(__name__)

def get_predict(predicting_type: str):
    if 'reconstruction' in predicting_type:
        return predict_reconstruction
    elif 'cross_entropy' in predicting_type:
        return predict_cross_entropy
    else:
        logger.warning(f"Unknown predicting type: {predicting_type}")
        return None


import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def predict_reconstruction(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader
) -> dict[str, float]:
    """
    Perform reconstruction using the provided model and data loader.
    Works with Sample dicts: UniModalSample and MultiModalSample.
    """
    model.eval()
    batch = next(iter(test_loader))

    # Extract inputs, labels, and mask
    X = batch["inputs"]
    y = X
    mask = batch.get("metadata", {}).get("mask") if batch.get("metadata") else None

    if mask is None:
        mask = torch.ones_like(y, dtype=torch.bool)

    # For reconstruction tasks, use inputs as target if label is None
    if y is None:
        y = X

    with torch.no_grad():
        y_pred = model(X)

    dis_accuracy = [] 
    cont_accuracy = [] 
    for column_idx in range(X.size(1)): 
        print(f"Feature Column {column_idx}:") 
        predicted_attributes = y_pred[column_idx] # shape: (batch_size, num_classes)
        targets = y[:, column_idx] # shape: (batch_size,)
        feature_mask = mask[:, column_idx] # shape: (batch_size,)
        predicted_attributes = predicted_attributes[feature_mask] 
        if predicted_attributes.size(1) > 1: 
            is_discrete = True 
            predicted_attributes = predicted_attributes.softmax(dim=-1).argmax(dim=-1).cpu().numpy() 
            if np.unique(predicted_attributes).size == 1: 
                logger.warning(f"Warning: Only one class predicted for feature column {column_idx}.") 
        else: 
            is_discrete = False 
            predicted_attributes = predicted_attributes.squeeze(-1).cpu().numpy()
        targets = targets[feature_mask].cpu().numpy()
        print(f"Predicted attribute values: {predicted_attributes[:5]}")
        print(f"True attribute values: {targets[:5]}")
        accuracy_value = accuracy(torch.tensor(predicted_attributes), torch.tensor(targets))
        print(f"Accuracy: {accuracy_value:.4f}\n")
        if is_discrete:
            dis_accuracy.append(accuracy_value)
        else:
            cont_accuracy.append(accuracy_value)

    results = {}
    accuracy_values = []
    if len(dis_accuracy) != 0:
        avg_dis_accuracy = np.mean(dis_accuracy)
        print(f"Average Discrete Accuracy: {avg_dis_accuracy:.4f}")
        accuracy_values += dis_accuracy
        results["avg_discrete_accuracy"] = avg_dis_accuracy
    if len(cont_accuracy) != 0:
        avg_cont_accuracy = np.mean(cont_accuracy)
        print(f"Average Continuous Accuracy: {avg_cont_accuracy:.4f}")
        accuracy_values += cont_accuracy
        results["avg_continuous_accuracy"] = avg_cont_accuracy
    avg_accuracy = np.mean(accuracy_values)
    print(f"Overall Average Accuracy: {avg_accuracy:.4f}")
    results["overall_avg_accuracy"] = avg_accuracy
    return results

def predict_cross_entropy(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader
) -> float:
    """
    Cross-entropy prediction for classification tasks.
    Works with Sample dicts.
    """
    batch = next(iter(test_loader))
    X = batch["inputs"]
    y = batch.get("label", None)

    if y is None:
        raise ValueError("Classification tasks require labels in 'label' field.")

    # For multi-modal inputs, pick first modality
    if isinstance(X, dict):
        modality = next(iter(X.keys()))
        X = X[modality]
        y = y[modality] if isinstance(y, dict) else y

    with torch.no_grad():
        y_pred = model(X)
        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]

    y_pred = y_pred.argmax(dim=1)

    # Compute accuracy
    correct = (y_pred == y).sum().item()
    acc = correct / y.size(0)

    return acc
