import sys
import os
import cv2
import torch

# Add the YOLOv5 path to sys.path
sys.path.append('C:/Users/Hein Htoo Naing/yolov5')  # Adjust the path as necessary

# Load the YOLOv5 model (choose the model variant: 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load the model

# Load an image
image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/input_image.jpg'))  # Path to input image
img = cv2.imread(image_path)

# Check if the image was loaded
if img is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    # Perform inference
    results = model(img)

    # Print results
    results.print()

    # Display the results on the image
    results.show()  # Optional: Show image with detections

    # Save results
    output_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/output_yolo_results.jpg'))
    results.save(output_image_path)

    print(f"Output image with detections saved as {output_image_path}")
