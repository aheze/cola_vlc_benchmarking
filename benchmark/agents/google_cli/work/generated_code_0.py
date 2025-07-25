import cv2
import numpy as np
import os
import json

def detect_coins(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return []

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Canny Edge Detection
    canny_edges = cv2.Canny(blurred, 50, 150)

    # Hough Circle Transform
    circles = cv2.HoughCircles(
        canny_edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=150
    )

    detected_coins = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            detected_coins.append({"x": int(x), "y": int(y), "radius": int(r)})

    return detected_coins

def main():
    input_dir = "/Users/andrew/Documents/Lab/cola_vlc_benchmarking/benchmark/tasks/coin_detection/input"
    output_file = "solution.json"
    
    # Assuming there's only one image in the input directory
    image_name = "three-centuries-of-american-coins_TCO_a_Main.jpg"
    image_path = os.path.join(input_dir, image_name)

    if not os.path.exists(image_path):
        print(f"Error: Input image not found at {image_path}")
        return

    coins = detect_coins(image_path)

    # Prepare the final JSON structure
    result = {"coins": coins}

    # Save the results to solution.json
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Coin detection complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()