#!/usr/bin/env python3
"""
Face Annotation Tool

A GUI tool for manually annotating faces in images and exporting the annotations
in the same JSON format as the face detection ground truth files.

Usage:
    python face_annotation_tool.py <image_path> [output_json_path]

Controls:
    - Click and drag to draw bounding boxes around faces
    - Press 'r' to remove the last bounding box
    - Press 'c' to clear all bounding boxes
    - Press 's' to save annotations to JSON file
    - Press 'q' or close window to quit
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from PIL import Image
import numpy as np


class FaceAnnotationTool:
    def __init__(self, image_path: str, output_path: Optional[str] = None):
        self.image_path = image_path
        self.output_path = output_path or self._get_default_output_path()
        
        # Load and validate image
        self.image = self._load_image()
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Initialize annotation data
        self.faces: List[Dict] = []
        self.current_box = None
        self.start_point = None
        
        # Setup matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(self.image)
        self.ax.set_title(f"Face Annotation Tool - {os.path.basename(image_path)}")
        
        # Setup event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Add buttons
        self._setup_buttons()
        
        # Instructions
        self._show_instructions()
        
    def _load_image(self) -> Optional[np.ndarray]:
        """Load image from file."""
        try:
            pil_image = Image.open(self.image_path)
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return np.array(pil_image)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def _get_default_output_path(self) -> str:
        """Generate default output path based on image path."""
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        return f"{base_name}_annotations.json"
    
    def _setup_buttons(self):
        """Setup GUI buttons."""
        # Create button axes
        ax_save = plt.axes([0.1, 0.02, 0.1, 0.04])
        ax_clear = plt.axes([0.25, 0.02, 0.1, 0.04])
        ax_remove = plt.axes([0.4, 0.02, 0.1, 0.04])
        ax_quit = plt.axes([0.8, 0.02, 0.1, 0.04])
        
        # Create buttons
        self.btn_save = Button(ax_save, 'Save (S)')
        self.btn_clear = Button(ax_clear, 'Clear (C)')
        self.btn_remove = Button(ax_remove, 'Remove (R)')
        self.btn_quit = Button(ax_quit, 'Quit (Q)')
        
        # Connect button events
        self.btn_save.on_clicked(lambda x: self.save_annotations())
        self.btn_clear.on_clicked(lambda x: self.clear_all_annotations())
        self.btn_remove.on_clicked(lambda x: self.remove_last_annotation())
        self.btn_quit.on_clicked(lambda x: plt.close())
    
    def _show_instructions(self):
        """Display instructions in the plot."""
        instructions = [
            "Instructions:",
            "• Click and drag to draw bounding boxes around faces",
            "• Press 'R' to remove last box, 'C' to clear all",
            "• Press 'S' to save, 'Q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            self.ax.text(0.02, 0.98 - i * 0.03, instruction, 
                        transform=self.ax.transAxes, fontsize=10,
                        verticalalignment='top', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def on_press(self, event):
        """Handle mouse press events."""
        if event.inaxes != self.ax:
            return
        
        self.start_point = (event.xdata, event.ydata)
        
        # Create new rectangle patch
        self.current_box = patches.Rectangle(
            self.start_point, 0, 0,
            linewidth=2, edgecolor='red', facecolor='none', alpha=0.7
        )
        self.ax.add_patch(self.current_box)
    
    def on_motion(self, event):
        """Handle mouse motion events (dragging)."""
        if self.current_box is None or self.start_point is None:
            return
        
        if event.inaxes != self.ax:
            return
        
        # Update rectangle size
        width = event.xdata - self.start_point[0]
        height = event.ydata - self.start_point[1]
        
        self.current_box.set_width(width)
        self.current_box.set_height(height)
        
        self.fig.canvas.draw()
    
    def on_release(self, event):
        """Handle mouse release events."""
        if self.current_box is None or self.start_point is None:
            return
        
        if event.inaxes != self.ax:
            return
        
        # Calculate final bounding box
        x1, y1 = self.start_point
        x2, y2 = event.xdata, event.ydata
        
        # Ensure positive width/height
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # Minimum size check
        if width < 10 or height < 10:
            self.current_box.remove()
            self.current_box = None
            self.start_point = None
            self.fig.canvas.draw()
            return
        
        # Clamp to image boundaries
        x = max(0, x)
        y = max(0, y)
        width = min(width, self.image.shape[1] - x)
        height = min(height, self.image.shape[0] - y)
        
        # Update rectangle to final position
        self.current_box.set_xy((x, y))
        self.current_box.set_width(width)
        self.current_box.set_height(height)
        
        # Calculate center and area
        center_x = x + width / 2
        center_y = y + height / 2
        area = width * height
        
        # Create face annotation
        face_annotation = {
            "face_id": len(self.faces) + 1,
            "bounding_box": {
                "x": int(round(x)),
                "y": int(round(y)),
                "width": int(round(width)),
                "height": int(round(height))
            },
            "center": {
                "x": int(round(center_x)),
                "y": int(round(center_y))
            },
            "area": int(round(area))
        }
        
        self.faces.append(face_annotation)
        
        # Add face ID label
        self.ax.text(x, y - 5, f"Face {face_annotation['face_id']}", 
                    color='red', fontsize=10, fontweight='bold')
        
        # Reset for next annotation
        self.current_box = None
        self.start_point = None
        
        self.fig.canvas.draw()
        print(f"Added face {face_annotation['face_id']}: {face_annotation['bounding_box']}")
    
    def on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == 'r':
            self.remove_last_annotation()
        elif event.key == 'c':
            self.clear_all_annotations()
        elif event.key == 's':
            self.save_annotations()
        elif event.key == 'q':
            plt.close()
    
    def remove_last_annotation(self):
        """Remove the last annotated face."""
        if not self.faces:
            print("No annotations to remove")
            return
        
        # Remove last face from list
        removed_face = self.faces.pop()
        print(f"Removed face {removed_face['face_id']}")
        
        # Redraw image and remaining annotations
        self._redraw_annotations()
    
    def clear_all_annotations(self):
        """Clear all annotations."""
        self.faces.clear()
        print("Cleared all annotations")
        self._redraw_annotations()
    
    def _redraw_annotations(self):
        """Redraw the image and all current annotations."""
        self.ax.clear()
        self.ax.imshow(self.image)
        self.ax.set_title(f"Face Annotation Tool - {os.path.basename(self.image_path)}")
        
        # Redraw all face annotations
        for face in self.faces:
            bbox = face['bounding_box']
            rect = patches.Rectangle(
                (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                linewidth=2, edgecolor='red', facecolor='none', alpha=0.7
            )
            self.ax.add_patch(rect)
            
            # Add face ID label
            self.ax.text(bbox['x'], bbox['y'] - 5, f"Face {face['face_id']}", 
                        color='red', fontsize=10, fontweight='bold')
        
        self._show_instructions()
        self.fig.canvas.draw()
    
    def save_annotations(self):
        """Save annotations to JSON file."""
        if not self.faces:
            print("No annotations to save")
            return
        
        # Get image dimensions
        height, width = self.image.shape[:2]
        channels = self.image.shape[2] if len(self.image.shape) > 2 else 1
        
        # Create annotation data structure
        annotation_data = {
            "image_path": os.path.basename(self.image_path),
            "image_dimensions": {
                "width": width,
                "height": height,
                "channels": channels
            },
            "total_faces": len(self.faces),
            "faces": self.faces
        }
        
        # Save to JSON file
        try:
            with open(self.output_path, 'w') as f:
                json.dump(annotation_data, f, indent=2)
            print(f"Saved {len(self.faces)} face annotations to {self.output_path}")
        except Exception as e:
            print(f"Error saving annotations: {e}")
    
    def run(self):
        """Start the annotation tool."""
        print(f"Starting face annotation tool for: {self.image_path}")
        print(f"Output will be saved to: {self.output_path}")
        print("\nControls:")
        print("  - Click and drag to draw bounding boxes around faces")
        print("  - Press 'R' to remove last box, 'C' to clear all")
        print("  - Press 'S' to save, 'Q' to quit")
        
        plt.show()
        
        # Auto-save if there are annotations when closing
        if self.faces:
            save_prompt = input(f"\nSave {len(self.faces)} annotations to {self.output_path}? (y/n): ")
            if save_prompt.lower().startswith('y'):
                self.save_annotations()


def main():
    parser = argparse.ArgumentParser(description="Face Annotation Tool")
    parser.add_argument("image_path", help="Path to the image file to annotate")
    parser.add_argument("-o", "--output", help="Output JSON file path", default=None)
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    try:
        # Create and run annotation tool
        tool = FaceAnnotationTool(args.image_path, args.output)
        tool.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 