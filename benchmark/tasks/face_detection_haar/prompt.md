## Task Description
Your goal is to detect ALL faces in the provided test images. Do this by using the haar-cascade filters with image and post-processing as needed.
You cannot have any false-positives or missed detections.

There are 3 test images in the input folder that need to be processed.

## Output Requirements
Save your final detections for faces in a json file named `solution.json`.
The structure should be an array of results, one for each image, with the same keys as the following json example (where x,y for the bounding box dictionary is the top, left most coordinate of the box. x is the horizontal axis and y is the vertical axis of the image):

```json
[
  {
    "image_path": "face_test_1.jpg",
    "image_dimensions": {
      "width": 683,
      "height": 630,
      "channels": 3
    },
    "total_faces": 2,
    "faces": [
      {
        "face_id": 1,
        "bounding_box": {
          "x": 350,
          "y": 450,
          "width": 100,
          "height": 100
        },
        "center": {
          "x": 400,
          "y": 500
        },
        "area": 10000
      },
      {
        "face_id": 2,
        "bounding_box": {
          "x": 630,
          "y": 100,
          "width": 150,
          "height": 100
        },
        "center": {
          "x": 705,
          "y": 150
        },
        "area": 15000
      }
    ]
  },
  {
    "image_path": "face_test_2.jpg",
    "image_dimensions": {
      "width": 2000,
      "height": 1333,
      "channels": 3
    },
    "total_faces": 1,
    "faces": [
      {
        "face_id": 1,
        "bounding_box": {
          "x": 800,
          "y": 200,
          "width": 120,
          "height": 120
        },
        "center": {
          "x": 860,
          "y": 260
        },
        "area": 14400
      }
    ]
  }
]
```