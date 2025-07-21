import json
import time
import os
from typing import Any, Dict, List, Tuple
from datetime import datetime
from benchmark.core.base_evaluator import BaseEvaluator
from benchmark.core.result_types import EvaluationResult

class FaceDetectionEvaluator(BaseEvaluator):
    """
    Evaluator for the Face Detection Task using Haar cascades.
    Compares detected faces with ground truth using IoU metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.iou_threshold = config.get("evaluation_criteria", {}).get("iou_threshold", 0.5)
        self.f1_threshold = config.get("evaluation_criteria", {}).get("f1_threshold", 0.9)
        
        self.print_task_info()

    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1, box2: Dictionaries with keys 'x', 'y', 'width', 'height'
            
        Returns:
            IoU score as float between 0 and 1
        """
        # Extract coordinates
        x1, y1, w1, h1 = box1['x'], box1['y'], box1['width'], box1['height']
        x2, y2, w2, h2 = box2['x'], box2['y'], box2['width'], box2['height']
        
        # Calculate intersection coordinates
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        # Check if there's an intersection
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area
        
        # Avoid division by zero
        if union_area == 0:
            return 0.0
            
        return intersection_area / union_area

    def _load_json_file(self, file_path: str) -> Dict:
        """Load and parse JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON file {file_path}: {e}")
            return None

    def _validate_json_structure(self, data, file_type: str) -> bool:
        """Validate the structure of the JSON data."""
        # Handle both single image (dict) and multiple images (list) formats
        if isinstance(data, dict):
            # Single image format
            images_data = [data]
        elif isinstance(data, list):
            # Multiple images format
            images_data = data
        else:
            print(f"Invalid data type in {file_type}: expected dict or list")
            return False
        
        # Validate each image
        for img_idx, img_data in enumerate(images_data):
            required_fields = ['image_path', 'image_dimensions', 'total_faces', 'faces']
            
            for field in required_fields:
                if field not in img_data:
                    print(f"Missing required field '{field}' in image {img_idx} of {file_type}")
                    return False
            
            # Validate faces structure
            for i, face in enumerate(img_data['faces']):
                required_face_fields = ['face_id', 'bounding_box', 'center', 'area']
                for field in required_face_fields:
                    if field not in face:
                        print(f"Missing required field '{field}' in face {i} of image {img_idx} in {file_type}")
                        return False
                
                # Validate bounding_box structure
                bbox_fields = ['x', 'y', 'width', 'height']
                for field in bbox_fields:
                    if field not in face['bounding_box']:
                        print(f"Missing required field '{field}' in bounding_box of face {i} in image {img_idx} of {file_type}")
                        return False
        
        return True

    def _match_faces(self, predicted_faces: List[Dict], ground_truth_faces: List[Dict]) -> Tuple[List[Tuple], List[int], List[int]]:
        """
        Match predicted faces with ground truth faces using IoU.
        
        Returns:
            - List of (pred_idx, gt_idx, iou_score) tuples for matches above threshold
            - List of unmatched predicted face indices
            - List of unmatched ground truth face indices
        """
        matches = []
        used_gt = set()
        used_pred = set()
        
        # Calculate IoU for all pairs and find best matches
        for pred_idx, pred_face in enumerate(predicted_faces):
            best_match = None
            best_iou = 0
            
            for gt_idx, gt_face in enumerate(ground_truth_faces):
                if gt_idx in used_gt:
                    continue
                    
                iou = self._calculate_iou(pred_face['bounding_box'], gt_face['bounding_box'])
                if iou >= self.iou_threshold and iou > best_iou:
                    best_match = gt_idx
                    best_iou = iou
            
            if best_match is not None:
                matches.append((pred_idx, best_match, best_iou))
                used_gt.add(best_match)
                used_pred.add(pred_idx)
        
        # Find unmatched faces
        unmatched_pred = [i for i in range(len(predicted_faces)) if i not in used_pred]
        unmatched_gt = [i for i in range(len(ground_truth_faces)) if i not in used_gt]
        
        return matches, unmatched_pred, unmatched_gt

    def evaluate(self, solution_folder: str, solution_config: Any = None) -> EvaluationResult:
        """
        Evaluate face detection by comparing solution with ground truth.
        """
        start_time = time.time()
        
        # Get file paths
        solution_file_name = self.config["expected_outputs"]["solution_file"]
        ground_truth_file_name = self.config["expected_outputs"]["ground_truth_file"]
        
        solution_path = os.path.join(solution_folder, solution_file_name)
        # Ground truth is in the task directory
        task_dir = os.path.dirname(os.path.abspath(__file__))
        ground_truth_path = os.path.join(task_dir, ground_truth_file_name)
        
        print(f"Evaluating solution: {solution_path}")
        print(f"Against ground truth: {ground_truth_path}")
        
        # Check if files exist
        if not os.path.exists(solution_path):
            error_msg = f"Solution file not found: {solution_path}"
            return EvaluationResult(
                task_id=self.config.get("task_id", "face_detection_haar"),
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
                task_id=self.config.get("task_id", "face_detection_haar"),
                agent_id="unknown",
                timestamp=datetime.now(),
                metrics={},
                success=False,
                execution_time=time.time() - start_time,
                error_message=error_msg,
                artifacts={}
            )

        # Load JSON files
        solution_data = self._load_json_file(solution_path)
        ground_truth_data = self._load_json_file(ground_truth_path)
        
        if solution_data is None:
            error_msg = f"Failed to load solution JSON: {solution_path}"
            return EvaluationResult(
                task_id=self.config.get("task_id", "face_detection_haar"),
                agent_id="unknown",
                timestamp=datetime.now(),
                metrics={},
                success=False,
                execution_time=time.time() - start_time,
                error_message=error_msg,
                artifacts={}
            )
            
        if ground_truth_data is None:
            error_msg = f"Failed to load ground truth JSON: {ground_truth_path}"
            return EvaluationResult(
                task_id=self.config.get("task_id", "face_detection_haar"),
                agent_id="unknown",
                timestamp=datetime.now(),
                metrics={},
                success=False,
                execution_time=time.time() - start_time,
                error_message=error_msg,
                artifacts={}
            )

        # Validate JSON structure
        if not self._validate_json_structure(solution_data, "solution"):
            error_msg = "Invalid solution JSON structure"
            return EvaluationResult(
                task_id=self.config.get("task_id", "face_detection_haar"),
                agent_id="unknown",
                timestamp=datetime.now(),
                metrics={},
                success=False,
                execution_time=time.time() - start_time,
                error_message=error_msg,
                artifacts={}
            )

        if not self._validate_json_structure(ground_truth_data, "ground truth"):
            error_msg = "Invalid ground truth JSON structure"
            return EvaluationResult(
                task_id=self.config.get("task_id", "face_detection_haar"),
                agent_id="unknown",
                timestamp=datetime.now(),
                metrics={},
                success=False,
                execution_time=time.time() - start_time,
                error_message=error_msg,
                artifacts={}
            )

        # Handle both single image and multiple images formats
        if isinstance(solution_data, dict):
            solution_images = [solution_data]
        else:
            solution_images = solution_data
            
        if isinstance(ground_truth_data, dict):
            ground_truth_images = [ground_truth_data]
        else:
            ground_truth_images = ground_truth_data
        
        # Validate we have matching number of images
        if len(solution_images) != len(ground_truth_images):
            error_msg = f"Mismatch in number of images: solution has {len(solution_images)}, ground truth has {len(ground_truth_images)}"
            return EvaluationResult(
                task_id=self.config.get("task_id", "face_detection_haar"),
                agent_id="unknown",
                timestamp=datetime.now(),
                metrics={},
                success=False,
                execution_time=time.time() - start_time,
                error_message=error_msg,
                artifacts={}
            )
        
        # Evaluate each image and collect results
        image_results = []
        total_matches = 0
        total_predicted = 0
        total_ground_truth = 0
        total_false_positives = 0
        total_false_negatives = 0
        total_iou_sum = 0
        total_perfect_detections = 0
        all_matches = []
        
        for img_idx, (solution_img, gt_img) in enumerate(zip(solution_images, ground_truth_images)):
            # Validate image paths match (optional check)
            if solution_img.get('image_path') != gt_img.get('image_path'):
                print(f"Warning: Image path mismatch for image {img_idx}: solution='{solution_img.get('image_path')}' vs gt='{gt_img.get('image_path')}'")
            
            # Extract faces for this image
            predicted_faces = solution_img['faces']
            ground_truth_faces = gt_img['faces']
            
            # Match faces using IoU
            matches, unmatched_pred, unmatched_gt = self._match_faces(predicted_faces, ground_truth_faces)
            
            # Calculate metrics for this image
            num_predicted = len(predicted_faces)
            num_ground_truth = len(ground_truth_faces)
            num_matches = len(matches)
            
            # Image-level metrics
            precision = num_matches / num_predicted if num_predicted > 0 else 0.0
            recall = num_matches / num_ground_truth if num_ground_truth > 0 else 0.0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            avg_iou = sum(match[2] for match in matches) / len(matches) if matches else 0.0
            perfect_detection = (num_matches == num_ground_truth) and (num_predicted == num_ground_truth)
            
            # Check if total_faces matches
            reported_total = solution_img.get('total_faces', 0)
            actual_total = len(predicted_faces)
            total_faces_match = reported_total == actual_total
            
            # Store image result
            image_result = {
                "image_path": gt_img.get('image_path', f"image_{img_idx}"),
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "average_iou": avg_iou,
                "num_predicted": num_predicted,
                "num_ground_truth": num_ground_truth,
                "num_matches": num_matches,
                "num_false_positives": len(unmatched_pred),
                "num_false_negatives": len(unmatched_gt),
                "perfect_detection": perfect_detection,
                "total_faces_reported": reported_total,
                "total_faces_actual": actual_total,
                "total_faces_match": total_faces_match
            }
            image_results.append(image_result)
            
            # Accumulate totals
            total_matches += num_matches
            total_predicted += num_predicted
            total_ground_truth += num_ground_truth
            total_false_positives += len(unmatched_pred)
            total_false_negatives += len(unmatched_gt)
            total_iou_sum += avg_iou * num_matches if matches else 0
            if perfect_detection:
                total_perfect_detections += 1
            all_matches.extend(matches)
        
        # Calculate overall metrics
        overall_precision = total_matches / total_predicted if total_predicted > 0 else 0.0
        overall_recall = total_matches / total_ground_truth if total_ground_truth > 0 else 0.0
        overall_f1_score = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        overall_avg_iou = total_iou_sum / total_matches if total_matches > 0 else 0.0
        overall_perfect_detection = total_perfect_detections == len(image_results)
        
        # Success based on F1 score threshold from config
        success = overall_perfect_detection or (overall_f1_score >= self.f1_threshold)
        
        # Prepare metrics
        metrics = {
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1_score": overall_f1_score,
            "overall_average_iou": overall_avg_iou,
            "total_predicted": total_predicted,
            "total_ground_truth": total_ground_truth,
            "total_matches": total_matches,
            "total_false_positives": total_false_positives,
            "total_false_negatives": total_false_negatives,
            "overall_perfect_detection": overall_perfect_detection,
            "num_images": len(image_results),
            "perfect_images": total_perfect_detections,
            "iou_threshold": self.iou_threshold,
            "f1_threshold": self.f1_threshold,
            "image_results": image_results
        }
        
        exec_time = time.time() - start_time
        
        return EvaluationResult(
            task_id=self.config.get("task_id", "face_detection_haar"),
            agent_id="unknown",
            timestamp=datetime.now(),
            metrics=metrics,
            success=success,
            execution_time=exec_time,
            error_message=None,
            artifacts={"image_results": image_results, "all_matches": all_matches}
        )

    def get_metrics(self) -> List[str]:
        return [
            "overall_precision", "overall_recall", "overall_f1_score", "overall_average_iou", 
            "total_predicted", "total_ground_truth", "total_matches",
            "total_false_positives", "total_false_negatives", "overall_perfect_detection",
            "num_images", "perfect_images"
        ]

    def generate_report(self, results: List[EvaluationResult]) -> str:
        lines = []
        lines.append("=== Face Detection Evaluation Report ===")
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
            
            # Overall Results Summary
            lines.append("Overall Detection Results:")
            lines.append(f"  Images Evaluated: {metrics.get('num_images', 0)}")
            lines.append(f"  Perfect Images: {metrics.get('perfect_images', 0)}")
            lines.append(f"  Total Faces in Ground Truth: {metrics.get('total_ground_truth', 0)}")
            lines.append(f"  Total Faces Detected: {metrics.get('total_predicted', 0)}")
            lines.append(f"  Total Correct Matches: {metrics.get('total_matches', 0)}")
            lines.append(f"  Total False Positives: {metrics.get('total_false_positives', 0)}")
            lines.append(f"  Total False Negatives: {metrics.get('total_false_negatives', 0)}")
            lines.append("")
            
            lines.append("Overall Performance Metrics:")
            lines.append(f"  Precision: {metrics.get('overall_precision', 0):.4f}")
            lines.append(f"  Recall: {metrics.get('overall_recall', 0):.4f}")
            lines.append(f"  F1-Score: {metrics.get('overall_f1_score', 0):.4f} (threshold: >= {metrics.get('f1_threshold', 0):.2f})")
            lines.append(f"  Average IoU: {metrics.get('overall_average_iou', 0):.4f}")
            lines.append(f"  IoU Threshold: {metrics.get('iou_threshold', 0):.2f}")
            lines.append("")
            
            lines.append("Overall Status:")
            lines.append(f"  Perfect Detection (All Images): {metrics.get('overall_perfect_detection', False)}")
            lines.append("")
            
            # Per-Image Results
            image_results = metrics.get('image_results', [])
            if image_results:
                lines.append("Per-Image Results:")
                for i, img_result in enumerate(image_results):
                    lines.append(f"  Image {i+1}: {img_result.get('image_path', 'Unknown')}")
                    lines.append(f"    Faces GT/Detected/Matched: {img_result.get('num_ground_truth', 0)}/{img_result.get('num_predicted', 0)}/{img_result.get('num_matches', 0)}")
                    lines.append(f"    Precision/Recall/F1: {img_result.get('precision', 0):.3f}/{img_result.get('recall', 0):.3f}/{img_result.get('f1_score', 0):.3f}")
                    lines.append(f"    Perfect Detection: {img_result.get('perfect_detection', False)}")
                    if not img_result.get('total_faces_match', True):
                        lines.append(f"    ⚠ Total faces mismatch: reported {img_result.get('total_faces_reported', 0)}, actual {img_result.get('total_faces_actual', 0)}")
                    lines.append("")
            
            # Interpretation
            f1_score = metrics.get('overall_f1_score', 0)
            f1_threshold = metrics.get('f1_threshold', 0.9)
            
            if f1_score >= f1_threshold:
                performance = "Passed (meets F1 threshold)"
            elif f1_score >= f1_threshold * 0.9:
                performance = "Good (close to threshold)"
            elif f1_score >= f1_threshold * 0.7:
                performance = "Fair"
            else:
                performance = "Poor"
                
            lines.append(f"  Overall Performance: {performance}")
            
            if metrics.get('overall_perfect_detection', False):
                lines.append("  ✓ Perfect detection achieved across all images!")
            elif metrics.get('perfect_images', 0) > 0:
                lines.append(f"  ✓ Perfect detection on {metrics.get('perfect_images', 0)} out of {metrics.get('num_images', 0)} images")
            
            if metrics.get('total_false_positives', 0) > 0:
                lines.append(f"  ⚠ {metrics.get('total_false_positives', 0)} total false positive(s) detected")
            if metrics.get('total_false_negatives', 0) > 0:
                lines.append(f"  ⚠ {metrics.get('total_false_negatives', 0)} total face(s) missed")
            
            lines.append("")
            lines.append("-" * 50)
            lines.append("")
        
        return "\n".join(lines) 