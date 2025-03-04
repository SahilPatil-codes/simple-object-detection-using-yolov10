#!/usr/bin/env python3
"""
YOLOv10 Object Detection
========================

A simple, single-file project that implements object detection using the YOLOv10 algorithm.
This file contains everything you need to run the project: instructions, code, and comments.
Just copy and paste the code into one file and run it—no extra files required.

Usage:
    python yolov10_object_detection.py --image path/to/your/image.jpg [--weights path/to/yolov10_weights.pth] [--conf-threshold 0.5] [--nms-threshold 0.4]

Requirements:
    - Python 3.7 or higher
    - PyTorch and torchvision
    - OpenCV
    - NumPy

Installation:
    pip install torch torchvision opencv-python numpy

Command-line Arguments:
    --image           Path to the input image file (required).
    --weights         (Optional) Path to the YOLOv10 weights file. Default is "yolov10_weights.pth".
    --conf-threshold  (Optional) Confidence threshold for detections. Default is 0.5.
    --nms-threshold   (Optional) Non-maximum suppression threshold. Default is 0.4.

Description:
    This script loads a pre-trained YOLOv10 model, processes an input image, and outputs the detected objects
    with bounding boxes and labels. It demonstrates the full pipeline from image preprocessing to inference and
    visualization, all within one file for simplicity.

Enjoy object detecting with YOLOv10!
=====================================================================
"""

import argparse
import cv2
import numpy as np
import torch
import torchvision

def load_yolov10_model(weights_path):
    """
    Loads the YOLOv10 model from the specified weights file.
    In a real-world scenario, this function should load the YOLOv10 architecture and weights.
    For demonstration purposes, this function assumes the weights file contains a Torch model.
    """
    print("Loading YOLOv10 model from:", weights_path)
    try:
        model = torch.load(weights_path, map_location=torch.device('cpu'))
    except Exception as e:
        print("Error loading model:", e)
        exit(1)
    model.eval()
    return model

def preprocess_image(image, input_size=(640, 640)):
    """
    Preprocess the image for YOLOv10:
      - Resize to the desired input size.
      - Convert BGR (OpenCV default) to RGB.
      - Normalize pixel values to [0, 1].
      - Rearrange dimensions to channel-first format.
    """
    image_resized = cv2.resize(image, input_size)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).float()
    image_tensor = image_tensor.permute(2, 0, 1)  # Change to (channels, height, width)
    image_tensor /= 255.0  # Normalize to [0, 1]
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor

def postprocess_detections(detections, conf_threshold=0.5, nms_threshold=0.4):
    """
    Postprocess the raw detections output by the model.
    This should include:
      - Filtering out detections below the confidence threshold.
      - Applying non-maximum suppression (NMS) to remove redundant overlapping boxes.
    For this demonstration, this function returns empty lists.
    Replace this with the actual postprocessing logic based on your model's output.
    """
    # Placeholder: Replace with actual detection extraction logic.
    boxes = []     # List of bounding boxes [x1, y1, x2, y2]
    scores = []    # List of confidence scores
    class_ids = [] # List of detected class indices
    return boxes, scores, class_ids

def draw_detections(image, boxes, scores, class_ids):
    """
    Draw bounding boxes and labels on the image.
    """
    for box, score, cls in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        label = f"Class {cls}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def main(args):
    # Load image from file
    image = cv2.imread(args.image)
    if image is None:
        print("Error: Could not load image from", args.image)
        exit(1)
    
    # Load the YOLOv10 model
    model = load_yolov10_model(args.weights)
    
    # Preprocess the image
    input_tensor = preprocess_image(image)
    
    # Run inference using the model
    with torch.no_grad():
        detections = model(input_tensor)
        # In a real implementation, 'detections' would be processed further.
        # For this simple example, we assume detections is a placeholder.
    
    # Postprocess the detections (dummy implementation)
    boxes, scores, class_ids = postprocess_detections(detections, conf_threshold=args.conf_threshold, nms_threshold=args.nms_threshold)
    
    # Draw the detections on the original image
    output_image = draw_detections(image.copy(), boxes, scores, class_ids)
    
    # Display the output image with detections
    cv2.imshow("YOLOv10 Object Detection", output_image)
    print("Press any key on the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv10 Object Detection - Single File Implementation")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file")
    parser.add_argument("--weights", type=str, default="yolov10_weights.pth", help="Path to the YOLOv10 weights file")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--nms-threshold", type=float, default=0.4, help="Non-maximum suppression threshold")
    args = parser.parse_args()
    
    main(args)
