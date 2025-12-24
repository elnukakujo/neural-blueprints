from .train import get_train_inference
from .eval import get_eval_inference
from .predict import get_predict_inference

__all__ = [
    get_train_inference,
    get_eval_inference,
    get_predict_inference,
]