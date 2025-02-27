import cv2
from screeninfo import get_monitors
import torch
from ultralytics import YOLO



# Colors: Console
white = '\033[38;2;255;255;255m'
silver = '\033[38;2;204;204;204m'
purple = '\033[38;2;189;22;255m'
magenta = '\033[38;2;255;0;128m'
pink = '\033[38;2;255;0;242m'
cyan = '\033[38;2;22;255;212m'
green = '\033[38;2;5;232;49m'
greenLight = '\033[38;2;204;255;188m'
greenDark = '\033[38;2;30;121;13m'
yellow = '\033[38;2;255;217;24m'
orange = '\033[38;2;247;151;31m'
red = '\033[91m'
resetColor = '\033[0m'

# Set device
if torch.cuda.is_available():
    device = 'cuda:0'
    print(f'Using device:{magenta} {device}{resetColor}\n'
          f'Device Name:{magenta} {torch.cuda.get_device_name(device)}'
          f'{resetColor}\n')
else:
    import platform
    device = 'cpu'
    print(f'Using device:{magenta} {device}{resetColor}\n'
          f'Device Name:{magenta} {platform.processor()}'
          f'{f'{resetColor}'}\n')



def webcam():
    # Get the dimensions of the primary monitor and define window size
    monitor = get_monitors()[0]
    windowScale = 0.6 # Adjust window size relative to the screen size
    width, height = int(monitor.width * windowScale), int(monitor.height * windowScale)


    # Instantiating a pre-trained YOLOv8n model
    model = YOLO('yolov8n.pt') # 's' is the smallest model; use 'l' or 'x' for larger ones
    model.to(device)


    # Open a connection to the webcam (0 is usually the built-in webcam)
    video = cv2.VideoCapture(0)

    if not video.isOpened():
        print('Error: Could not open webcam.')
        return

    # Loop to continuously capture frames
    while True:
        # Capture each frame
        ret, frame = video.read()

        # Check if frame is captured
        if not ret:
            print('Error: Could not read frame.')
            break

        # Perform object detection on the frame
        results = model(frame, verbose=False)

        # Draw bounding boxes and labels on the frame
        frameAnnotated = results[0].plot() # Annotate the frame with detections
        frameAnnotated = cv2.resize(frameAnnotated, (width, height))

        # Display the annotated frame in a window
        cv2.imshow('Webcam Feed', frameAnnotated)


        # Exit the loop if the 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key
            break

    # Release the webcam and close windows
    video.release()
    cv2.destroyAllWindows()

webcam()
