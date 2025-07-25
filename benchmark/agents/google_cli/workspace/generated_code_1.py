import cv2
import json
import os
import urllib.request

def detect_faces_in_image(image_path, cascade_classifier):
    """
    Detects faces in a single image, returning structured data.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        return None

    height, width, channels = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Parameters for detectMultiScale. These may need tuning for specific images.
    # scaleFactor: How much the image size is reduced at each image scale. 1.05 is a good value.
    # minNeighbors: How many neighbors each candidate rectangle should have to retain it.
    # minSize: Minimum possible object size.
    faces = cascade_classifier.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    face_data = []
    for i, (x, y, w, h) in enumerate(faces):
        face_id = i + 1
        bounding_box = {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        }
        center = {
            "x": int(x + w / 2),
            "y": int(y + h / 2)
        }
        area = int(w * h)

        face_data.append({
            "face_id": face_id,
            "bounding_box": bounding_box,
            "center": center,
            "area": area
        })

    return {
        "image_path": os.path.basename(image_path),
        "image_dimensions": {
            "width": width,
            "height": height,
            "channels": channels
        },
        "total_faces": len(faces),
        "faces": face_data
    }

def main():
    """
    Main function to process all images in the input directory and save results.
    """
    input_dir = 'input'
    output_file = 'solution.json'
    cascade_filename = 'haarcascade_frontalface_default.xml'

    # --- Haar Cascade File Handling ---
    # Check for the cascade file in the current directory first.
    if os.path.exists(cascade_filename):
        haar_cascade_path = cascade_filename
    # If not found, check the cv2 data path.
    elif os.path.exists(os.path.join(cv2.data.haarcascades, cascade_filename)):
        haar_cascade_path = os.path.join(cv2.data.haarcascades, cascade_filename)
    # If still not found, download it.
    else:
        print(f"'{cascade_filename}' not found. Attempting to download...")
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{cascade_filename}"
        try:
            urllib.request.urlretrieve(url, cascade_filename)
            haar_cascade_path = cascade_filename
            if not os.path.exists(haar_cascade_path):
                print("Error: Download failed. Cannot proceed without the cascade file.")
                return
            print("Download successful.")
        except Exception as e:
            print(f"An error occurred during download: {e}")
            return

    face_cascade = cv2.CascadeClassifier(haar_cascade_path)

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        # Create an empty solution file if the input directory is missing
        with open(output_file, 'w') as f:
            json.dump([], f, indent=2)
        return

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"Warning: No images found in the '{input_dir}' directory.")
        with open(output_file, 'w') as f:
            json.dump([], f, indent=2)
        return

    all_results = []
    # Sort files to ensure consistent order
    for image_file in sorted(image_files):
        image_path = os.path.join(input_dir, image_file)
        result = detect_faces_in_image(image_path, face_cascade)
        if result:
            all_results.append(result)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Processing complete. Detections saved to {output_file}")

if __name__ == '__main__':
    main()