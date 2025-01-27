import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random
import sys
import torch
from torchvision import models, transforms



# ========================================== User Inputs =========================================
# Input 1: File Location
inImagePath = 'img_6.png'

# Input 2: Nural Network
inModelType = 'ResNet'

# Input 3: Print Options
inPrintNumber = 10



# ======================================= Define Parameters ======================================
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
if torch.cuda.is_available():
    device = 'cuda:0'
    print('\n====================================== Set Training Device '
          '=====================================')
    print(f'Train with Device:{magenta} {device}{resetColor}\n'
          f'Device Name:{magenta} {torch.cuda.get_device_name(device)}{resetColor}\n\n')
else:
    import platform
    device = 'cpu'
    print('\n====================================== Set Training Device '
          '=====================================')
    print(f'Train with Device:{magenta} {device}{resetColor}\n'
          f'Device Name:{magenta} {platform.processor()}{resetColor}\n')


# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256x256
    transforms.CenterCrop(224),  # Center crop to 224x224 (the input size for ResNet)
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) # Normalize with ImageNet stats
    ])



# ======================================= Define Functions =======================================
def pressKey(event):
    if event.key == 'escape':
        plt.close()



def openImage():
    # Open the image
    image = Image.open(inImagePath)

    # Display the image
    image.show()



def plotImage():
    # Read the image
    image = mpimg.imread(inImagePath)

    # Create a figure with a specified size and face color
    fig = plt.figure(figsize=(10, 8), facecolor='black')

    # Display the image
    plt.imshow(image)
    plt.axis("off")  # Hide axes for a cleaner view

    # Position the figure
    manager = plt.get_current_fig_manager()
    manager.window.geometry(f"+{700}+{100}")

    fig.canvas.mpl_connect('key_press_event', pressKey)
    plt.show()



def loadClassLabels(modelType):
    print('=================================== Load: Class Labels'
          ' ==================================')
    # Load: Class labels
    if modelType == 'ResNet':
        with open('labelsResNet.txt') as file:
            # Clean the labels by stripping whitespace, quotes, and commas
            labels = [label.strip().replace('"', '').replace(',', '')
                      for label in file.readlines()]
    elif modelType == 'Custom':
        with open('classes.txt') as file:
            labels = [line.strip() for line in file.readlines()]


    # Print: Class labels
    print(f'Classes:{purple} {modelType}{greenLight}')
    numLabels = len(labels)
    if numLabels > inPrintNumber:
        indices = []
        for _ in range(inPrintNumber):
            indexRandom = random.randint(0, numLabels-1)
            while indexRandom in indices:
                indexRandom = random.randint(0, numLabels-1)
            indices.append(indexRandom)
            print(f'     {labels[indexRandom]}')
        print(f'     ...{resetColor}\n')
    else:
        for index, label in enumerate(labels):
            print(f'     {labels[index]}')
            if index >= inPrintNumber:
                print('     ...{resetColor}\n')
                break
    print(f'Number of Classes:{white} {len(labels)}{resetColor}\n\n')

    return labels



def importModel(modelType):
    print('==================================== Importing Model '
          '====================================')
    print(f'Model Type:{purple} {modelType}{resetColor}\n')

    if modelType == 'ResNet':
        from torchvision.models import ResNet50_Weights

        # Load the pre-trained ResNet model (ResNet50 in this example)
        loadedModel = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        if device != 'cpu':
            loadedModel.to(device)
        print(f'Model is on device:{magenta} {next(loadedModel.parameters()).device}'
              f'{resetColor}\n\n')

        # Set the model to evaluation mode
        loadedModel.eval()

        return loadedModel
    else:
        raise ValueError(f'{orange}ERROR: Unsupported model type.\n'
                         f'Please choose a different name.\n\n')


# Load and preprocess the image
def preprocessImage(img):
    img = Image.open(img) # Load the image
    img = preprocess(img) # Apply the transformations
    img = img.unsqueeze(0) # Add a batch dimension

    return img



def evaluateImage(nn, img):
    print('===================================== Evaluate Image '
          '====================================')
    # Make predictions
    with torch.no_grad():
        output = nn(img)

    # Process the output
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    topKProbabilities, topKIndices = torch.topk(probabilities, k=len(labels))

    # Print the results
    print(f'Top{pink} {inPrintNumber}{resetColor} Predictions:')
    for index in range(topKProbabilities.size(0)):
        print(f'     {labels[topKIndices[index]]}:'
              f'{red} {topKProbabilities[index].item() * 100:.3f}%{resetColor}')
        if index >= inPrintNumber:
            break
    print('\n')



# ========================================= Run The Code =========================================
# Load: Model
model = importModel(modelType=inModelType)

# Load: Labels
labels = loadClassLabels(modelType=inModelType)

# Preprocess & evaluate image
image = preprocessImage(img=inImagePath).to(device)

# Make predictions
evaluateImage(nn=model, img=image)

# Plot: Image
userInput = input(f'Do you want to see the picture? (y/n) ')
print('')
if userInput == 'Y' or userInput == 'y':
    # plotImage()
    openImage()

