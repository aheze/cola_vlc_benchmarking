from abc import ABC, abstractmethod
from typing import Any, Dict, List
from .result_types import EvaluationResult

class BaseEvaluator(ABC):
    """
    Abstract base class for all task evaluators.
    Each evaluator must implement this interface.
    """
    
    def print_task_info(self):
        """
        Prints the task name and description.
        """
        self.task_name = self.config.get("name", "Unnamed Task")
        self.task_description = self.config.get("description", "No description provided.")
        print(f"Initializing evaluator for task: {self.task_name}")
        print(f"Description: {self.task_description}")

    @abstractmethod
    def evaluate(self, solution_path: str, ground_truth: Any) -> EvaluationResult:
        """
        Evaluates the agent's solution against the ground truth.
        Returns an EvaluationResult object.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> List[str]:
        """
        Returns a list of metric names used in evaluation.
        """
        pass

    @abstractmethod
    def generate_report(self, results: List[EvaluationResult]) -> str:
        """
        Generates a human-readable report from a list of evaluation results.
        """
        pass
