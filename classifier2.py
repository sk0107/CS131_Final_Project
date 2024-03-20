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

import json

train_dataset_x, train_dataset_y = None, None
test_dataset_x, test_dataset_y = None, None

print('dataset load start')

train_dataset_x = torch.load('train_dataset_x.pth')
train_dataset_y = torch.load('train_dataset_y.pth')
test_dataset_x = torch.load('test_dataset_x.pth')
test_dataset_y = torch.load('test_dataset_y.pth')

print('dataset load done')

from tensorflow import keras

print('import keras done')

from keras.models import load_model
import numpy as np
import cv2 as cv

print('imports quarter done')

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, activations
from keras.utils import np_utils
from keras import backend as K

print('imports halfway done')

K.set_image_dim_ordering('th')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import os
from imutils import paths
import glob

target_size = (200, 200)

class ImageFolderDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(100, 100), split='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))
        self.classes.remove('.DS_Store')
        for label in self.classes:
            label_dir = os.path.join(data_dir, label)
            for image_file in os.listdir(label_dir):
                image_file_ext = image_file[image_file.rfind('.') + 1:]
                valid_file_exts = ['jpg', 'jpeg', 'png', 'gif']
                if image_file_ext in valid_file_exts:
                    image_path = os.path.join(label_dir, image_file)
                    self.image_paths.append(image_path)
                    self.labels.append(self.classes.index(label))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.target_size)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

print('imports done')

def cnn_model():
    model = Sequential() 
    model.add(Conv2D(32, (3, 3), input_shape=(3, target_size[0], target_size[1]), activation = 'relu')) 
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # the CONV CONV POOL structure is popularized in during ImageNet 2014
    model.add(Dropout(0.25)) # this thing called dropout is used to prevent overfitting
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
   
    model.add(Flatten()) 
    
    model.add(Dropout(0.5))
    
    model.add(Dense(4, activation= 'softmax'))
    
    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    return model

print("[INFO] creating model...")
model = cnn_model()
# Fit the model
print("[INFO] training model...")
records = model.fit(train_dataset_x, train_dataset_y, validation_split=0.1, epochs=25, batch_size=16, verbose=1)
# Final evaluation of the model
print("[INFO] evaluating model...")
scores = model.evaluate(test_dataset_x, test_dataset_y, verbose=1)
print('Final CNN accuracy: ', scores[1])

print('Finished Training')

PATH = 'classfier2.h5'

print("[INFO] saving model...")
model.save(PATH)

import matplotlib.pyplot as plt
cnn_probab = model.predict(test_dataset_x, batch_size=32, verbose=0)

# extract the probability for the label that was predicted:
p_max = np.amax(cnn_probab, axis=1)

plt.hist(p_max, normed=True, bins=list(np.linspace(0,1,11)))
plt.xlabel('p of predicted class')

N = 25
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), records.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), records.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), records.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), records.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

