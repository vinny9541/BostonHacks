import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

# data set 
garbageDF = 'garbage classification'

# recycle or trash
classes = os.listdir(garbageDF)
print(classes)

# grab the data from folders & resize them to be the same size
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


dataset = ImageFolder(garbageDF, transform = transformations)

# graph it 
import matplotlib.pyplot as plt
torch.manual_seed(42)

trainDS, valDS, testDS = random_split(dataset, [1593, 176, 758])
print(len(trainDS), len(valDS), len(testDS))

from torch.utils.data.dataloader import DataLoader
batch_size = 200

train_dl = DataLoader(trainDS, batch_size, shuffle = True)
val_dl = DataLoader(valDS, batch_size*2)


# show each bach
from torchvision.utils import make_grid

def display_batch(data_loader):
    for images, labels in data_loader:
        for i in range(images.shape[0]):
            image = transforms.ToPILImage()(images[i])  
            label_idX = labels[i].item()
            folder_name = classes[label_idX]

            # Display the image with label as title
            plt.imshow(image)
            plt.title(f"Label: {folder_name}")
            plt.show()

# Display batches for training and validation
display_batch(train_dl)
display_batch(val_dl)







