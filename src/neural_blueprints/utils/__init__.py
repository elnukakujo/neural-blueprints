"""Neural Network Backbones Library"""
from .activation import get_activation
from .blocks import get_block
from .trainer import Trainer
from .device import get_device
from .visualize import curve_plot, bar_plot, heatmap_plot, image_plot
from .metrics import accuracy, recall, f1_score
from .criterion import mse, rmse, mae, cross_entropy, binary_cross_entropy, vae_loss, get_criterion
from .optimizer import get_optimizer
from .types import infer_types
from .get_projection import get_input_projection, get_output_projection

__version__ = "0.1.0"
__all__ = [
    get_activation,
    get_block,
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
    rmse,
    cross_entropy,
    binary_cross_entropy,
    vae_loss,
    get_criterion,
    get_optimizer,
    infer_types,
    get_input_projection,
    get_output_projection,
]