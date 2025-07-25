import cv2
import os
import json

def detect_faces_in_images():
    """
    Detects faces in images using Haar Cascade classifiers, processes the results,
    and saves them to a JSON file.
    """
    # --- Configuration ---
    # Assume the input images are in a directory named 'input' in the current working directory.
    input_dir = 'input'
    output_json_path = 'solution.json'
    
    # Path to the Haar Cascade XML file for frontal face detection.
    # Using cv2.data.haarcascades to get the path to the pre-trained models.
    try:
        casc_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(casc_path):
            raise FileNotFoundError(f"Haar Cascade file not found at {casc_path}")
    except Exception as e:
        print(f"Error finding Haar Cascade file: {e}")
        print("Please ensure OpenCV is installed correctly or provide a valid path to 'haarcascade_frontalface_default.xml'.")
        return

    # Create the face cascade classifier
    face_cascade = cv2.CascadeClassifier(casc_path)

    # --- Directory and File Handling ---
    # Create the input directory if it doesn't exist.
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created directory: '{input_dir}'")
        print("Please place your test images (face_test_1.jpg, face_test_2.jpg, etc.) inside this directory.")
        # Since the script cannot proceed without images, we will exit.
        return

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in the '{input_dir}' directory. Please add images to process.")
        return
        
    print(f"Found {len(image_files)} images to process: {image_files}")

    # --- Main Processing Loop ---
    all_results = []

    for image_path in sorted(image_files):
        full_image_path = os.path.join(input_dir, image_path)
        
        # Read the image
        image = cv2.imread(full_image_path)
        if image is None:
            print(f"Warning: Could not read image {full_image_path}. Skipping.")
            continue

        # Get image dimensions
        height, width, channels = image.shape
        image_dims = {"width": width, "height": height, "channels": channels}

        # Convert to grayscale for the detector
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Improve contrast - this can improve detection accuracy
        gray_image = cv2.equalizeHist(gray_image)

        # --- Face Detection ---
        # The detectMultiScale function detects objects and returns a list of rectangles.
        # scaleFactor: How much the image size is reduced at each image scale. 1.1 is a good value.
        # minNeighbors: How many neighbors each candidate rectangle should have to retain it.
        #               Higher values result in fewer detections but with higher quality. This helps with false positives.
        # minSize: Minimum possible object size. Objects smaller than this are ignored.
        faces_detected = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        print(f"Processing '{image_path}': Found {len(faces_detected)} faces.")

        # --- Result Formatting ---
        face_entries = []
        for i, (x, y, w, h) in enumerate(faces_detected):
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

            face_entries.append({
                "face_id": face_id,
                "bounding_box": bounding_box,
                "center": center,
                "area": area
            })

        image_result = {
            "image_path": image_path,
            "image_dimensions": image_dims,
            "total_faces": len(faces_detected),
            "faces": face_entries
        }
        all_results.append(image_result)

    # --- Save Final JSON Output ---
    try:
        with open(output_json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Successfully saved detection results to '{output_json_path}'")
    except IOError as e:
        print(f"Error writing to JSON file: {e}")

if __name__ == '__main__':
    detect_faces_in_images()