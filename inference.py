import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

import os
from PIL import Image
from torch.utils.data import Dataset

class ImageFolderDataset(Dataset):
    def __init__(self, data_dir, input_dir, transform=None, target_size=(100, 100), split='train'):
        self.data_dir = input_dir
        self.transform = transform
        self.target_size = target_size
        self.image_paths = []
        for image_file in os.listdir(input_dir):
            image_file_ext = image_file[image_file.rfind('.') + 1:]
            valid_file_exts = ['jpg', 'jpeg', 'png', 'gif', 'webp']
            if image_file_ext in valid_file_exts:
                image_path = os.path.join(input_dir, image_file)
                self.image_paths.append(image_path)
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))
        self.classes.remove('.DS_Store')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.target_size)
        if self.transform:
            image = self.transform(image)
        return image
    
dataset_dir = 'dataset/'
input_dir = 'inputs/'
inputs = ImageFolderDataset(dataset_dir, input_dir, transform=transform)
classes = inputs.classes

print(classes)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 25)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 20, 25)
        # self.size_set = False
        self.fc1 = nn.Linear(20 * 7 * 7, 432)
        self.fc2 = nn.Linear(432, 200)
        self.fc3 = nn.Linear(200, 142)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # if not self.size_set:
        #     self.size_set = True
        #     self.fc1 = nn.Linear(x.size(dim=1), 432)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

PATH = 'classfier1.pth'
net = Net()
net.load_state_dict(torch.load(PATH))

from torch.utils.data import DataLoader

inputloader = DataLoader(inputs, batch_size=1)

with torch.no_grad():
    for data in inputloader:
        images = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        print(classes[predicted[0]])
        # total += labels.size(0)
        # correct += (predicted == labels).sum().item()