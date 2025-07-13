import json
import time
import os
from typing import Any, Dict, List
from datetime import datetime
from benchmark.core.base_evaluator import BaseEvaluator
from benchmark.core.result_types import EvaluationResult

# Image evaluation imports
import torch
import torchvision.transforms as transforms
from PIL import Image
import lpips
import clip

class HomographyEstimationEvaluator(BaseEvaluator):
    """
    Evaluator for the Homography Estimation Task.
    Compares stitched panorama to ground truth using LPIPS and CLIP metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lpips_threshold = config.get("evaluation_criteria", {}).get("lpips_threshold", 0.3)
        self.clip_threshold = config.get("evaluation_criteria", {}).get("clip_threshold", 0.95)
        self.require_both_metrics = config.get("evaluation_criteria", {}).get("require_both_metrics", True)
        
        # Initialize models
        self._lpips_model = None
        self._clip_model = None
        self._clip_preprocess = None
        
        self.print_task_info()

    def _load_models(self):
        """Lazy loading of LPIPS and CLIP models."""
        if self._lpips_model is None:
            self._lpips_model = lpips.LPIPS(net='alex')
            
        if self._clip_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=device)

    def _load_and_preprocess_image(self, image_path: str, target_size=None):
        """Load and preprocess image for evaluation."""
        try:
            image = Image.open(image_path).convert('RGB')
            if target_size:
                image = image.resize(target_size, Image.LANCZOS)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def _calculate_lpips_score(self, img1_path: str, img2_path: str):
        """Calculate LPIPS score between two images."""
        try:
            self._load_models()
            
            # Load images
            img1 = self._load_and_preprocess_image(img1_path)
            img2 = self._load_and_preprocess_image(img2_path)
            
            if img1 is None or img2 is None:
                return None
                
            # Resize images to same size if needed
            if img1.size != img2.size:
                min_width = min(img1.size[0], img2.size[0])
                min_height = min(img1.size[1], img2.size[1])
                target_size = (min_width, min_height)
                img1 = img1.resize(target_size, Image.LANCZOS)
                img2 = img2.resize(target_size, Image.LANCZOS)
            
            # Convert to tensors
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            tensor1 = transform(img1).unsqueeze(0)
            tensor2 = transform(img2).unsqueeze(0)
            
            # Calculate LPIPS
            with torch.no_grad():
                lpips_score = self._lpips_model(tensor1, tensor2)
                
            return lpips_score.item()
            
        except Exception as e:
            print(f"Error calculating LPIPS: {e}")
            return None

    def _calculate_clip_score(self, img1_path: str, img2_path: str):
        """Calculate CLIP similarity score between two images."""
        try:
            self._load_models()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load and preprocess images
            img1 = self._load_and_preprocess_image(img1_path)
            img2 = self._load_and_preprocess_image(img2_path)
            
            if img1 is None or img2 is None:
                return None
                
            # Preprocess for CLIP
            img1_tensor = self._clip_preprocess(img1).unsqueeze(0).to(device)
            img2_tensor = self._clip_preprocess(img2).unsqueeze(0).to(device)
            
            # Get image features
            with torch.no_grad():
                img1_features = self._clip_model.encode_image(img1_tensor)
                img2_features = self._clip_model.encode_image(img2_tensor)
                
                # Normalize features
                img1_features = img1_features / img1_features.norm(dim=-1, keepdim=True)
                img2_features = img2_features / img2_features.norm(dim=-1, keepdim=True)
                
                # Calculate cosine similarity
                clip_score = torch.sum(img1_features * img2_features, dim=-1)
                
            return clip_score.item()
            
        except Exception as e:
            print(f"Error calculating CLIP score: {e}")
            return None

    def evaluate(self, solution_folder: str, solution_config: Any = None) -> EvaluationResult:
        """
        Evaluate homography estimation by comparing solution panorama with ground truth.
        """
        start_time = time.time()
        
        # Get file paths
        solution_file_name = self.config["expected_outputs"]["solution_file"]
        ground_truth_file_name = self.config["expected_outputs"]["ground_truth_file"]
        
        solution_path = os.path.join(solution_folder, solution_file_name)
        # Ground truth is in the task directory
        task_dir = os.path.dirname(os.path.abspath(__file__))
        ground_truth_path = os.path.join(task_dir, ground_truth_file_name)
        print("Ground truth path: ", ground_truth_path)
        
        print(f"Evaluating solution: {solution_path}")
        print(f"Against ground truth: {ground_truth_path}")
        
        # Check if files exist
        if not os.path.exists(solution_path):
            error_msg = f"Solution file not found: {solution_path}"
            return EvaluationResult(
                task_id=self.config.get("task_id", "homography_estimation"),
                agent_id="unknown",
                timestamp=datetime.now(),
                metrics={},
                success=False,
                execution_time=time.time() - start_time,
                error_message=error_msg,
                artifacts={}
            )
            
        if not os.path.exists(ground_truth_path):
            error_msg = f"Ground truth file not found: {ground_truth_path}"
            return EvaluationResult(
                task_id=self.config.get("task_id", "homography_estimation"),
                agent_id="unknown",
                timestamp=datetime.now(),
                metrics={},
                success=False,
                execution_time=time.time() - start_time,
                error_message=error_msg,
                artifacts={}
            )

        # Calculate metrics
        print("Calculating LPIPS score...")
        lpips_score = self._calculate_lpips_score(solution_path, ground_truth_path)
        
        print("Calculating CLIP score...")
        clip_score = self._calculate_clip_score(solution_path, ground_truth_path)
        
        # Determine success
        lpips_pass = lpips_score is not None and lpips_score < self.lpips_threshold
        clip_pass = clip_score is not None and clip_score > self.clip_threshold
        
        if self.require_both_metrics:
            success = lpips_pass and clip_pass
        else:
            success = lpips_pass or clip_pass
            
        # Prepare metrics
        metrics = {
            "lpips_score": lpips_score if lpips_score is not None else -1,
            "clip_score": clip_score if clip_score is not None else -1,
            "lpips_threshold": self.lpips_threshold,
            "clip_threshold": self.clip_threshold,
            "lpips_pass": lpips_pass,
            "clip_pass": clip_pass,
            "both_required": self.require_both_metrics
        }
        
        exec_time = time.time() - start_time
        
        return EvaluationResult(
            task_id=self.config.get("task_id", "homography_estimation"),
            agent_id="unknown",
            timestamp=datetime.now(),
            metrics=metrics,
            success=success,
            execution_time=exec_time,
            error_message=None,
            artifacts={}
        )

    def get_metrics(self) -> List[str]:
        return ["lpips_score", "clip_score", "lpips_pass", "clip_pass", "lpips_threshold", "clip_threshold"]

    def generate_report(self, results: List[EvaluationResult]) -> str:
        lines = []
        lines.append("=== Homography Estimation Evaluation Report ===")
        lines.append("")
        
        for res in results:
            lines.append(f"Task: {res.task_id}, Agent: {res.agent_id}")
            lines.append(f"Success: {res.success}")
            lines.append(f"Execution Time: {res.execution_time:.2f}s")
            lines.append("")
            
            if res.error_message:
                lines.append(f"Error: {res.error_message}")
                lines.append("")
                continue
                
            metrics = res.metrics
            lines.append("Metrics:")
            lines.append(f"  LPIPS Score: {metrics.get('lpips_score', 'N/A'):.4f} (threshold: < {metrics.get('lpips_threshold', 'N/A')})")
            lines.append(f"  CLIP Score: {metrics.get('clip_score', 'N/A'):.4f} (threshold: > {metrics.get('clip_threshold', 'N/A')})")
            lines.append(f"  LPIPS Pass: {metrics.get('lpips_pass', False)}")
            lines.append(f"  CLIP Pass: {metrics.get('clip_pass', False)}")
            
            if metrics.get('both_required', True):
                lines.append("  Requirement: Both LPIPS and CLIP must pass")
            else:
                lines.append("  Requirement: Either LPIPS or CLIP must pass")
                
            lines.append("")
            
            # Interpretation
            lpips_score = metrics.get('lpips_score', -1)
            clip_score = metrics.get('clip_score', -1)
            
            if lpips_score >= 0:
                if lpips_score < 0.1:
                    lpips_quality = "Excellent"
                elif lpips_score < 0.3:
                    lpips_quality = "Good"
                elif lpips_score < 0.5:
                    lpips_quality = "Fair"
                else:
                    lpips_quality = "Poor"
                lines.append(f"  Perceptual similarity: {lpips_quality}")
            
            if clip_score >= 0:
                if clip_score > 0.95:
                    clip_quality = "Excellent"
                elif clip_score > 0.9:
                    clip_quality = "Good"
                elif clip_score > 0.8:
                    clip_quality = "Fair"
                else:
                    clip_quality = "Poor"
                lines.append(f"  Semantic similarity: {clip_quality}")
            
            lines.append("")
            lines.append("-" * 50)
            lines.append("")
        
        return "\n".join(lines) 