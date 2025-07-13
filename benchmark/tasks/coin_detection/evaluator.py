import json
import time
from typing import Any, Dict, List
from datetime import datetime
from benchmark.core.base_evaluator import BaseEvaluator
from benchmark.core.result_types import EvaluationResult
import math
import os

class CoinDetectionEvaluator(BaseEvaluator):
    """
    Evaluator for the Coin Detection Task.
    Compares detected coins to ground truth using count and spatial accuracy.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.position_tolerance = config.get("evaluation_criteria", {}).get("position_tolerance", 10)
        self.radius_tolerance = config.get("evaluation_criteria", {}).get("radius_tolerance", 0.15)
        self.print_task_info()  # Print task info during initialization

    def evaluate(self, solution_folder: str, solution_config: Any = None) -> EvaluationResult:
        """
        If solution_config is None, attempt to load from config['expected_outputs']['sample'].
        """
        start_time = time.time()
        solution_file_name = self.config["expected_outputs"]["solution_file"]
        print(solution_file_name)
        solution_path = os.path.join(solution_folder, solution_file_name)
        with open(solution_path, "r") as f:
            pred = json.load(f)["coins"]
        # Use provided solution_config, or fall back to config sample
        if solution_config is not None:
            gt = solution_config["coins"]
        else:
            gt = self.config.get("expected_outputs", {}).get("sample", {}).get("coins", [])

        matched = 0
        used_gt = set()
        for p in pred:
            for i, g in enumerate(gt):
                if i in used_gt:
                    continue
                dist = math.hypot(p["x"] - g["x"], p["y"] - g["y"])
                radius_close = abs(p["radius"] - g["radius"]) <= self.radius_tolerance * g["radius"]
                if dist <= self.position_tolerance and radius_close:
                    matched += 1
                    used_gt.add(i)
                    break

        precision = matched / len(pred) if pred else 0.0
        recall = matched / len(gt) if gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        success = (f1 == 1.0)

        metrics = {
            "matched": matched,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_pred": len(pred),
            "total_gt": len(gt)
        }
        exec_time = time.time() - start_time
        return EvaluationResult(
            task_id=self.config.get("task_id", "coin_detection"),
            agent_id="unknown",
            timestamp=datetime.now(),
            metrics=metrics,
            success=success,
            execution_time=exec_time,
            error_message=None,
            artifacts={}
        )

    def get_metrics(self) -> List[str]:
        return ["matched", "precision", "recall", "f1", "total_pred", "total_gt"]

    def generate_report(self, results: List[EvaluationResult]) -> str:
        lines = []
        for res in results:
            lines.append(f"Task: {res.task_id}, Agent: {res.agent_id}, Success: {res.success}")
            for k, v in res.metrics.items():
                lines.append(f"  {k}: {v}")
            if res.error_message:
                lines.append(f"  Error: {res.error_message}")
            lines.append("")
        return "\n".join(lines)
