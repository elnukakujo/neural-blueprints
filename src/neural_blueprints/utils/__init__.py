"""Neural Network Backbones Library"""
from .activation import get_activation
from .blocks import get_block
from .normalization import get_normalization
from .trainer import Trainer
from .device import get_device
from .visualize import curve_plot, bar_plot, heatmap_plot, image_plot
from .metrics import accuracy, recall, f1_score, mse, mae, root_mean_squared_error, r2_score, cross_entropy, binary_cross_entropy, vae_loss, get_criterion
from .types import infer_types

__version__ = "0.1.0"
__all__ = [
    get_activation,
    get_block,
    get_normalization,
    Trainer,
    get_device,
    curve_plot,
    bar_plot,
    heatmap_plot,
    image_plot,
    accuracy,
    recall,
    f1_score,
    mse,
    mae,
    root_mean_squared_error,
    r2_score,
    cross_entropy,
    binary_cross_entropy,
    vae_loss,
    get_criterion,
    infer_types,
]