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


def predict_reconstruction(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader
) -> dict[str, float]:
    """
    Perform reconstruction using the provided model and data loader.

    Args:
        model (torch.nn.Module): The neural network model for prediction.
        test_loader (torch.utils.data.DataLoader): DataLoader providing the input data.
    """
    model.eval()
    batch = next(iter(test_loader))

    if len(batch) == 3:
        X, y, mask = batch
    elif len(batch) == 2:
        X, y = batch
        mask = torch.ones_like(y, dtype=torch.bool)
    else:
        X = batch
        y = batch
        mask = torch.ones_like(y, dtype=torch.bool)
    assert isinstance(X, torch.Tensor), "Input data X must be a torch.Tensor"
    assert isinstance(y, torch.Tensor), "Target data y must be a torch.Tensor"
    assert isinstance(mask, torch.Tensor), "Mask data must be a torch.Tensor"

    assert X.dim() == 2, "Input data X must be a 2D tensor"
    assert y.dim() == 2, "Target data y must be a 2D tensor"
    assert mask.dim() == 2, "Mask data must be a 2D tensor"

    assert X.size() == y.size() == mask.size(), "Input data X, target data y, and mask must have the same shape"

    with torch.no_grad():
        y_pred = model(X)

    if isinstance(y_pred, tuple or list):
        y_pred = y_pred[0]

    dis_accuracy = []
    cont_accuracy = []
    for column_idx in range(X.size(1)):
        print(f"Feature Column {column_idx}:")
        predicted_attributes = y_pred[column_idx]      # shape: (batch_size, num_classes)
        targets = y[:, column_idx]                     # shape: (batch_size,)

        feature_mask = mask[:, column_idx]                  # shape: (batch_size,)
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
    X, y = next(iter(test_loader))

    assert isinstance(X, torch.Tensor), "Input data X must be a torch.Tensor"
    assert isinstance(y, torch.Tensor), "Target data y must be a torch.Tensor"

    assert X.dim() == 2, "Input data X must be a 2D tensor"
    assert y.dim() == 1, "Target data y must be a 1D tensor"

    with torch.no_grad():
        y_pred = model(X)
    
    if isinstance(y_pred, tuple or list):
        y_pred = y_pred[0]
    
    y_pred = y_pred.argmax(dim=1)
    if np.unique(y_pred).size == 1:
        logger.warning("Warning: Only one class predicted for this feature.")
    print(f"Predictions: {y_pred[:5]}, \n Ground Truth: {y[:5]}")
    acc = accuracy(y_pred, y)
    print(f"Prediction Accuracy: {acc:.4f}")
    return acc