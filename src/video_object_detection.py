import os
import cv2
import numpy as np

# Load the model configuration and weights
prototxt_path = "models/MobileNetSSD_deploy.prototxt"
caffemodel_path = "models/mobilenet_iter_73000.caffemodel"  # Make sure this matches your actual model file

# Check if the files exist
if not os.path.exists(prototxt_path):
    print(f"Error: Prototxt file not found at {prototxt_path}")
    exit()
if not os.path.exists(caffemodel_path):
    print(f"Error: Caffemodel file not found at {caffemodel_path}")
    exit()

# Load the Caffe model
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Define the list of classes MobileNet SSD was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]

# Open the video file or camera
video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../videos/classroom.mp4'))
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Prepare the frame for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Set the input to the model
    net.setInput(blob)

    # Perform the forward pass and get the detections
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak predictions
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and label on the frame
            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame with detections
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
