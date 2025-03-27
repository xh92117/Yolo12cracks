# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import DetectionPredictor
from .train import DetectionTrainer
from .val import DetectionValidator
from .model import DetectionModel

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator", "DetectionModel"
