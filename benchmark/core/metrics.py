from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

class BaseMetric(ABC):
    """
    Abstract base class for evaluation metrics.
    """

    @abstractmethod
    def compute(self, prediction: Any, ground_truth: Any) -> float:
        """
        Computes the metric value given prediction and ground truth.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the metric.
        """
        pass

class AccuracyMetric(BaseMetric):
    """
    Example accuracy metric for classification/detection tasks.
    """

    def compute(self, prediction: Any, ground_truth: Any) -> float:
        correct = 0
        total = len(ground_truth)
        for pred, gt in zip(prediction, ground_truth):
            if pred == gt:
                correct += 1
        return correct / total if total > 0 else 0.0

    def name(self) -> str:
        return "accuracy"

class PrecisionRecallMetric(BaseMetric):
    """
    Example precision/recall metric for detection tasks.
    """

    def compute(self, prediction: Any, ground_truth: Any) -> Tuple[float, float]:
        # Assumes prediction and ground_truth are sets of items
        pred_set = set(prediction)
        gt_set = set(ground_truth)
        true_positives = len(pred_set & gt_set)
        precision = true_positives / len(pred_set) if pred_set else 0.0
        recall = true_positives / len(gt_set) if gt_set else 0.0
        return precision, recall

    def name(self) -> str:
        return "precision_recall"
