import torch
import numpy as np
from ..metrics import accuracy

import logging
logger = logging.getLogger(__name__)

def get_predict_inference(predicting_type: str):
    if 'reconstruction' in predicting_type:
        return predict_reconstruction
    elif 'cross_entropy' in predicting_type:
        return predict_cross_entropy
    else:
        raise ValueError(f"Unknown predicting type: {predicting_type}")

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
    X, y, mask = next(iter(test_loader))

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
        logger.info(f"Feature Column {column_idx}:")
        predicted_attributes = y_pred[column_idx]      # shape: (batch_size, num_classes)
        targets = y[:, column_idx]                     # shape: (batch_size,)

        feature_mask = mask[:, column_idx]                  # shape: (batch_size,)
        predicted_attributes = predicted_attributes[feature_mask]
        if predicted_attributes.size(1) > 1:
            is_discrete = True
            predicted_attributes = predicted_attributes.softmax(dim=-1).argmax(dim=-1).cpu().numpy()
            if np.unique(predicted_attributes).size == 1:
                logger.warning("Warning: Only one class predicted for this feature.")
        else:
            is_discrete = False
            predicted_attributes = predicted_attributes.squeeze(-1).cpu().numpy()

        targets = targets[feature_mask].cpu().numpy()

        logger.info(f"Predicted attribute values: {predicted_attributes[:5]}")
        logger.info(f"True attribute values: {targets[:5]}")

        accuracy_value = accuracy(torch.tensor(predicted_attributes), torch.tensor(targets))
        logger.info(f"Accuracy: {accuracy_value:.4f}\n")
        if is_discrete:
            dis_accuracy.append(accuracy_value)
        else:
            cont_accuracy.append(accuracy_value)

    avg_dis_accuracy = np.mean(dis_accuracy)
    avg_cont_accuracy = np.mean(cont_accuracy)
    logger.info(f"Average Discrete Accuracy: {avg_dis_accuracy:.4f}")
    logger.info(f"Average Continuous Accuracy: {avg_cont_accuracy:.4f}")
    avg_accuracy = np.mean(dis_accuracy + cont_accuracy)
    logger.info(f"Overall Average Accuracy: {avg_accuracy:.4f}")

    return {
        "avg_discrete_accuracy": avg_dis_accuracy,
        "avg_continuous_accuracy": avg_cont_accuracy,
        "overall_avg_accuracy": avg_accuracy
    }

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
    logger.info(f"Predictions: {y_pred[:5]}, \n Ground Truth: {y[:5]}")
    acc = accuracy(y_pred, y)
    logger.info(f"Prediction Accuracy: {acc:.4f}")
    return acc