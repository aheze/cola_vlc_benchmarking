# Benchmark Tools

This directory contains utility tools for the benchmark suite.

## Face Annotation Tool

A GUI tool for manually annotating faces in images and exporting the annotations in the same JSON format as the face detection ground truth files.

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Basic usage:
```bash
python face_annotation_tool.py path/to/your/image.jpg
```

Specify custom output path:
```bash
python face_annotation_tool.py path/to/your/image.jpg -o custom_annotations.json
```

### Controls

- **Click and drag**: Draw bounding boxes around faces
- **R key**: Remove the last bounding box
- **C key**: Clear all bounding boxes  
- **S key**: Save annotations to JSON file
- **Q key**: Quit the application

### GUI Buttons

- **Save (S)**: Save current annotations
- **Clear (C)**: Clear all annotations
- **Remove (R)**: Remove last annotation
- **Quit (Q)**: Close the application

### Output Format

The tool outputs annotations in the same JSON format as the ground truth:

```json
{
  "image_path": "image.jpg",
  "image_dimensions": {
    "width": 683,
    "height": 630,
    "channels": 3
  },
  "total_faces": 3,
  "faces": [
    {
      "face_id": 1,
      "bounding_box": {
        "x": 48,
        "y": 115,
        "width": 69,
        "height": 69
      },
      "center": {
        "x": 82,
        "y": 149
      },
      "area": 4761
    }
  ]
}
```

### Features

- **Real-time visualization**: See bounding boxes as you draw them
- **Automatic calculations**: Center coordinates and area are calculated automatically
- **Input validation**: Minimum box size requirements and boundary clamping
- **Multiple formats**: Supports common image formats (JPEG, PNG, etc.)
- **Keyboard shortcuts**: Quick access to all functions
- **Auto-save prompt**: Prompts to save when closing with unsaved annotations

### Tips

1. Draw boxes slightly larger than the face for better coverage
2. Make sure the entire face (including hair/forehead) is enclosed
3. Use the 'R' key to quickly fix mistakes
4. Save frequently to avoid losing work
5. The tool automatically generates face IDs sequentially 