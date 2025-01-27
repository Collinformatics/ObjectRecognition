import random
import requests

# # URLs
# COCO:
    # 'https://raw.githubusercontent.com/kazuto1011/deeplab-pytorch/master/data/datasets/coco/labels.txt'

# ResNet:
    # 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'



# Save the file as:
savedFileName = 'ResNet'

# Correct URL for the raw labels file
url = ('https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json')

# Download the file
response = requests.get(url)
if response.status_code == 200:  # Check if the request was successful
    # Save the raw file
    with open(f'labels{savedFileName}.txt', "wb") as file:
        file.write(response.content)

    # Read the labels from the saved file
    with open(f'labels{savedFileName}.txt', "r") as file:
        labels = file.readlines()  # Read lines into a list

    # Clean the labels by stripping whitespace, quotes, commas, and numeric indices
    labels = [label.strip().replace('"', '').
              replace(',', '').split(maxsplit=1)[-1] for label in labels]

    # Save the cleaned labels without whitespace, quotes, commas, and numeric indices
    with open(f'labels{savedFileName}.txt', "w") as file:
        for label in labels:
            file.write(label + "\n")

    # Print labels
    numLabels = len(labels)
    print(f'Class Labels: {numLabels}')
    if numLabels > 10:
        indices = []
        for _ in range(10):  # Print 10 random labels
            indexRandom = random.randint(0,
                                         numLabels - 1)  # Adjust index to stay within bounds
            while indexRandom in indices:  # Avoid duplicates
                indexRandom = random.randint(0, numLabels - 1)
            indices.append(indexRandom)
            print(f'     {labels[indexRandom]}')
        print('     ...\n\n')
    else:
        for label in labels:
            print(f'     {label}')
        
    print(f'The downloaded labels were saved as labels{savedFileName}.txt '
          f'in the current working directory.')
else:
    print(f'Error: Unable to download the labels. Status code: {response.status_code}')
