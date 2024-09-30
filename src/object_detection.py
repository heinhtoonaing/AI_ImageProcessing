import cv2
import numpy as np
import os

# Paths to the model files
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
prototxt_path = os.path.join(base_path, 'MobileNetSSD_deploy.prototxt')
caffemodel_path = os.path.join(base_path, 'mobilenet_iter_73000.caffemodel')

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Define the classes for the object detection
classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog", 
           "horse", "motorbike", "person", "pottedplant", "sheep", 
           "sofa", "train", "tvmonitor"]

# Load the input image
# Load the input image
image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/input_image.jpg'))
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Failed to load image at {image_path}")
else:
    print(f"Image loaded successfully from {image_path}")

(h, w) = image.shape[:2]


# Create a blob from the image and perform a forward pass
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
net.setInput(blob)
detections = net.forward()

# Loop over the detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Filter out weak predictions
    if confidence > 0.2:
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw the prediction on the image
        label = f"{classes[idx]}: {confidence:.2f}"
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the output image
cv2.imwrite('C:/Users/Hein Htoo Naing/ai_image_processing_project/images/output_image.jpg', image)

print("Output image saved as output_image.jpg")
