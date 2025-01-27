import matplotlib.pyplot as plt
import random
import torch
from torchvision import models, transforms
from torchvision.models.detection import (fasterrcnn_resnet50_fpn, 
                                          FasterRCNN_ResNet50_FPN_Weights)
from PIL import Image, ImageDraw



# Input 1: File Location
inImagePath = 'birds.png'
inBoxColor = 'white'

# Input 2: Nural Network
inModelType = 'yolo'
inDetectionThreshold = 80
inFontSize = 30

# Input 3: Print Options
inPrintNumber = 10



# ================================== Define Parameters ===================================
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
# pd.set_option('display.float_format', '{:,.5f}'.format)

# Colors: Console
white = '\033[38;2;255;255;255m'
whiteA = white
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
redA = red
resetColor = '\033[0m'

# Set device
print('\n============================== Set Training Device '
          '==============================')
if torch.cuda.is_available():
    device = 'cuda:0'
    print(f'Train with Device:{magenta} {device}{resetColor}\n'
          f'Device Name:{magenta} {torch.cuda.get_device_name(device)}{resetColor}\n\n')
else:
    import platform
    device = 'cpu'
    print(f'Train with Device:{magenta} {device}{resetColor}\n'
          f'Device Name:{magenta} {platform.processor()}{resetColor}\n')


# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256x256
    transforms.CenterCrop(224),  # Center crop to 224x224
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) # Normalize with ImageNet stats
    ])



# =================================== Define Functions ===================================
def pressKey(event):
    if event.key == 'escape':
        plt.close()



def importModel(modelType):
    print('================================ Importing Model '
          '================================')
    print(f'Model Type:{purple} {modelType}{resetColor}\n')

    if modelType == 'COCO' or modelType == 'coco':
        from torchvision.models import ResNet50_Weights

        # Load the pre-trained model
        loadedModel = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f'{orange}ERROR: Unsupported model type.\n'
                         f'Please choose a different name.\n\n')


    # Set the model to evaluation mode
    loadedModel.eval()

    # Set device
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(f'Train with Device:{magenta} {device}{resetColor}\n'
              f'Device Name:{magenta} {torch.cuda.get_device_name(device)}'
              f'{resetColor}\n\n')
    else:
        device = 'cpu'
        print(f'Train with Device:{magenta} {device}{resetColor}\n\n')

    return loadedModel



def loadClassLabels(modelType):
    print('============================== Load: Class Labels '
          '===============================')
    print(f'Model Type:{purple} {modelType}{resetColor}\n')

    if modelType == 'COCO' or modelType == 'coco':
        with open('labelsCOCO.txt') as file:
            # Clean the labels by stripping whitespace, quotes, and commas
            labels = [label.strip() for label in file.readlines()]
    elif modelType == 'Custom':
        with open('classes.txt') as file:
            labels = [line.strip() for line in file.readlines()]
    else:
        raise ValueError(f'{orange}ERROR: Unsupported class label type.\n'
                         f'Please choose a different name.\n\n')

    # Print: Class labels
    print(f'Classes:{purple} {modelType}{greenLight}')
    numLabels = len(labels)
    if numLabels > inPrintNumber:
        indices = []
        for _ in range(inPrintNumber):
            indexRandom = random.randint(0, numLabels - 1)
            while indexRandom in indices:
                indexRandom = random.randint(0, numLabels - 1)
            indices.append(indexRandom)
            print(f'     {labels[indexRandom]}')
        print(f'     ...{resetColor}\n\n')
    else:
        for index, label in enumerate(labels):
            print(f'     {labels[index]}')
            if index >= inPrintNumber:
                print('     ...{resetColor}\n\n')
                break

    return labels



def detectObjectYOLO(path):
    import cv2
    global model
    from ultralytics import YOLO

    # Load the pre-trained model
    model = YOLO('yolov8l.pt') # 's' is the smallest model; use 'l' or 'x' for larger ones

    # Perform inference with just one line
    results = model(source=path, verbose=False)

    # Display the result image with OpenCV
    cv2.imshow('YOLOv8 Detection', results[0].plot()) # Draw detections
    cv2.waitKey(0) # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()



def detectObject(path):
    # Load the object detection model (Faster R-CNN)
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    model.to(device)


    # Define the image transformations (resize, normalize, etc.)
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert the image to a PyTorch tensor
    ])

    # Read the image
    img = Image.open(path).convert('RGB')

    # Preprocess the image
    imgPreprocessed = transform(img).unsqueeze(0).to(device)

    # Perform object detection
    with torch.no_grad():
        predictions = model(imgPreprocessed)[0]

    # Draw the bounding boxes around detected objects
    draw = ImageDraw.Draw(img)
    for index in range(len(predictions['boxes'])):
        score = predictions['scores'][index].item()
        score *= 100
        if score > inDetectionThreshold:
            detectedObject = labels[index]
            if detectedObject != 'background':
                # Get the bounding box coordinates
                box = predictions['boxes'][index].cpu().numpy()
                draw.rectangle(box.tolist(), outline=inBoxColor, width=3) # Draw the box
                # Optionally draw the label and score
                draw.text((box[0]+10, box[1]),
                          f'{detectedObject} {score:.2f} %',
                          fill=inBoxColor,
                          font_size=inFontSize)

    # Create a figure with a specified size and face color
    fig = plt.figure(figsize=(10, 8), facecolor='black')

    # Display the image with bounding boxes
    plt.imshow(img)
    plt.axis('off')  # Hide axes for a cleaner view

    # Position the figure
    manager = plt.get_current_fig_manager()
    manager.window.geometry(f'+{700}+{100}')

    fig.canvas.mpl_connect('key_press_event', pressKey)
    plt.show()



# ===================================== Run The Code =====================================
# Detect Object
if inModelType == 'YOLO' or inModelType == 'Yolo' or inModelType == 'yolo':
    detectObjectYOLO(path=inImagePath)
else:
    # Load: Model
    model = importModel(modelType=inModelType)

    # Load: Labels
    labels = loadClassLabels(modelType=inModelType)

    detectObject(path=inImagePath)
