import argparse
import os
import sys
import json

# Ensure project root is in sys.path for package imports
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Explicitly import all task modules to ensure registration
from benchmark.tasks.coin_detection.evaluator import CoinDetectionEvaluator
from benchmark.tasks.homography_estimation.evaluator import HomographyEstimationEvaluator

def main():
    parser = argparse.ArgumentParser(description="Run a vision coding agent benchmark task.")
    parser.add_argument("--task", type=str, required=True, help="Task ID to run (e.g., coin_detection)")
    parser.add_argument("--config", type=str, default=None, help="Path to task config.json (optional)")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to agent's solution workspace")
    args = parser.parse_args()

    # Determine config path
    config_path = args.config
    if not config_path:
        # Default: look for config.json in the task's directory
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "tasks", args.task, "config.json"
        )
        config_path = os.path.abspath(config_path)

    # Run evaluator (for coin_detection, use CoinDetectionEvaluator)
    print("-"* 40)
    with open(config_path, "r") as f:
        config = json.load(f)
    if args.task == "coin_detection":
        evaluator = CoinDetectionEvaluator(config)
        result = evaluator.evaluate(args.solution_path)
    elif args.task == "homography_estimation":
        evaluator = HomographyEstimationEvaluator(config)
        result = evaluator.evaluate(args.solution_path)
    else:
        print("No evaluator implemented for this task.")
        return
    
    if result.error_message:
        print("Error:", result.error_message)
    else:
        print("Evaluation metrics:", result.metrics)
        print("Success:", result.success)
    print("-"* 40)
if __name__ == "__main__":
    main()
