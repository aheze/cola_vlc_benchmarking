#!/usr/bin/env python3
"""
Demo script for the Face Annotation Tool

This script demonstrates how to use the face annotation tool with the test image
from the face detection task.
"""

import os
import sys

def main():
    # Get the path to the test image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_image_path = os.path.join(
        script_dir, "..", "tasks", "face_detection_haar", "input", "face_test_1.jpg"
    )
    test_image_path = os.path.abspath(test_image_path)
    
    # Check if the test image exists
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
        sys.exit(1)
    
    print("=== Face Annotation Tool Demo ===")
    print(f"Using test image: {test_image_path}")
    print()
    print("This will open the annotation tool with the face detection test image.")
    print("You can practice annotating the faces and save the results.")
    print()
    
    # Import and run the annotation tool
    try:
        from face_annotation_tool import FaceAnnotationTool
        
        # Create output path for demo
        output_path = os.path.join(script_dir, "demo_annotations.json")
        
        print(f"Annotations will be saved to: {output_path}")
        print()
        
        # Run the tool
        tool = FaceAnnotationTool(test_image_path, output_path)
        tool.run()
        
        print("Demo completed!")
        
    except ImportError as e:
        print(f"Error importing annotation tool: {e}")
        print("Make sure you have installed the requirements:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error running demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 