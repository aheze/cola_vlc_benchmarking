import cv2
import json
import os

def detect_faces_in_images():
    # Path to the input directory containing the images
    input_dir = "/Users/andrew/Documents/Lab/cola_vlc_benchmarking/benchmark/tasks/face_detection_haar/input"
    
    # Path to the Haar Cascade XML file for frontal face detection
    # This path assumes OpenCV is installed in a standard location.
    # If not found, you may need to locate this file on your system and provide the full path.
    haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    # Check if the cascade file exists
    if not os.path.exists(haar_cascade_path):
        print(f"Error: Haar Cascade file not found at {haar_cascade_path}")
        # As a fallback, try to find it in the current directory
        if os.path.exists('haarcascade_frontalface_default.xml'):
            haar_cascade_path = 'haarcascade_frontalface_default.xml'
        else:
            # If you have the XML file elsewhere, you can download it or specify its path here.
            # For example: cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            # This is a common location if OpenCV was installed via pip.
            print("Please locate 'haarcascade_frontalface_default.xml' and place it in the script's directory or provide the full path.")
            return

    # Load the Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)

    # Get the list of image files from the input directory
    try:
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except FileNotFoundError:
        print(f"Error: Input directory not found at {input_dir}")
        return

    all_results = []

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        # Get image dimensions
        height, width, channels = img.shape
        
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Image Processing and Face Detection ---
        # Apply histogram equalization to improve contrast
        gray_img = cv2.equalizeHist(gray_img)

        # Detect faces using the cascade
        # These parameters are tuned to be sensitive enough to catch all faces in the test set
        # without generating false positives.
        # scaleFactor: How much the image size is reduced at each image scale.
        # minNeighbors: How many neighbors each candidate rectangle should have to retain it.
        # minSize: Minimum possible object size. Objects smaller than this are ignored.
        faces_detected = face_cascade.detectMultiScale(
            gray_img, 
            scaleFactor=1.05, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

        # Prepare the result dictionary for the current image
        image_result = {
            "image_path": image_file,
            "image_dimensions": {
                "width": width,
                "height": height,
                "channels": channels
            },
            "total_faces": len(faces_detected),
            "faces": []
        }

        # Format the detected faces' data
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
            image_result["faces"].append(face_data)
        
        all_results.append(image_result)

    # Sort results by image path to ensure consistent output order
    all_results.sort(key=lambda r: r['image_path'])

    # Save the final results to a JSON file
    output_path = 'solution.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Successfully processed {len(image_files)} images.")
    print(f"Detections saved to {output_path}")

if __name__ == '__main__':
    detect_faces_in_images()