import cv2
import os
import json
from typing import List, Dict, Any, Tuple

def get_image_paths(input_dir: str) -> List[str]:
    """
    Gets a list of paths for supported image files in a directory.
    Supported extensions: .jpg, .jpeg, .png, .bmp, .tiff
    """
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    for filename in os.listdir(input_dir):
        if os.path.splitext(filename)[1].lower() in supported_extensions:
            image_paths.append(os.path.join(input_dir, filename))
    return image_paths

def detect_faces(image_path: str, face_cascade: cv2.CascadeClassifier) -> Dict[str, Any]:
    """
    Detects faces in a single image and returns data in the required format.
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image at {image_path}. Skipping.")
        return None

    # Get image dimensions
    height, width, channels = img.shape
    image_dimensions = {"width": width, "height": height, "channels": channels}

    # Convert to grayscale for the face detector
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # To meet the strict requirement of "no false-positives or missed detections",
    # these parameters for detectMultiScale are chosen carefully.
    # - scaleFactor=1.1: A smaller step for resizing the image, increasing the chance
    #   of finding faces of all sizes, but is computationally more expensive.
    # - minNeighbors=5: A higher value reduces false positives by requiring more
    #   confirmations for a detected face. This is a good balance.
    # - minSize=(30, 30): Ignores very small regions that are unlikely to be faces.
    faces_detected = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    face_results = []
    for i, (x, y, w, h) in enumerate(faces_detected):
        face_data = {
            "face_id": i + 1,
            "bounding_box": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            },
            "center": {
                "x": int(x + w / 2),
                "y": int(y + h / 2)
            },
            "area": int(w * h)
        }
        face_results.append(face_data)

    return {
        "image_path": os.path.basename(image_path),
        "image_dimensions": image_dimensions,
        "total_faces": len(face_results),
        "faces": face_results
    }

def main():
    """
    Main function to run the face detection process.
    """
    input_directory = "input"
    output_filename = "solution.json"
    
    # Verify input directory exists
    if not os.path.isdir(input_directory):
        print(f"Error: Input directory '{input_directory}' not found.")
        print("Please create it and place your test images inside.")
        # Create an empty JSON file as a placeholder
        with open(output_filename, 'w') as f:
            json.dump([], f, indent=2)
        return

    # Load the pre-trained Haar Cascade model for face detection
    # This path is robust and should work on most systems with OpenCV installed.
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    if not os.path.exists(cascade_path):
        print(f"Error: Haar Cascade file not found at {cascade_path}")
        print("Please ensure OpenCV is installed correctly.")
        return
        
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Get paths of images to process
    image_paths = get_image_paths(input_directory)
    if not image_paths:
        print(f"Warning: No images found in the '{input_directory}' directory.")
        with open(output_filename, 'w') as f:
            json.dump([], f, indent=2)
        return

    print(f"Found {len(image_paths)} images to process.")
    
    # Process each image and collect results
    all_results = []
    for image_path in sorted(image_paths): # Sort for consistent order
        print(f"Processing {image_path}...")
        result = detect_faces(image_path, face_cascade)
        if result:
            all_results.append(result)
    
    # Save the final results to a JSON file
    with open(output_filename, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDetection complete. Results saved to '{output_filename}'.")

if __name__ == "__main__":
    main()