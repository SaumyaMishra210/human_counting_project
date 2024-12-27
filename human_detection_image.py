import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLOv4 pre-trained model

# link for download yolov3.weights file
# https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights

# yolo_weights = '/kaggle/input/yolo/other/default/1/yolov3.weights'
# yolo_cfg = "/kaggle/input/yolo/other/default/1/yolov3.cfg"

yolo_weights = "models/yolov3.weights"
yolo_cfg = "models/yolov3.cfg"
yolo_net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

# Load class labels (COCO dataset)
file_path = "models/coco.names"
with open(file_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up input image (change the image path)
# image_path = "/kaggle/input/office-people/4.jpg"
image_path = "images/human.jpg"
image = cv2.imread(image_path)
height, width = image.shape[:2]

# Prepare image for YOLOv4 (resize and normalize)
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Set the blob as input to the network
yolo_net.setInput(blob)

# Get output layers
output_layers = yolo_net.getUnconnectedOutLayersNames()

# Run detection
layer_outputs = yolo_net.forward(output_layers)

# Initialize lists for detected objects
boxes = []
confidences = []
class_ids = []

# Loop through detections
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Only consider human (class ID = 0 for 'person')
        if confidence > 0.5 and class_id == 0:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Get coordinates for the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Store the box, confidence, and class ID
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression (NMS) to remove redundant boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Count the number of detected humans
human_count = len(indices.flatten())  # Number of humans detected

# Draw bounding boxes and label with counting numbers (1, 2, 3, ...)
for idx, i in enumerate(indices.flatten()):
    x, y, w, h = boxes[i]
    label = str(idx + 1)  # Sequential count (1, 2, 3, ...)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display the human count on the image
cv2.putText(image, f"Human Count: {human_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Convert BGR image to RGB for displaying with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the result using matplotlib
plt.imshow(image_rgb)
plt.axis('off')  # Hide axes
plt.show()
