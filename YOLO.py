import cv2
from ultralytics import YOLO


# Path to your image
image = 'supra.png'

# Instantiating a pre-trained YOLOv8n model
model = YOLO('yolov8l.pt')
    # 's' is the smallest model; use 'l' or 'x' for larger ones


# Perform inference with just one line
results = model(source=image)

# Display the result image with OpenCV
cv2.imshow('YOLOv8 Detection', results[0].plot())  # Draw detections on the image
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()
