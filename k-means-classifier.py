import torch
import torchvision
import torchvision.transforms as transforms

import os
from PIL import Image, ImageStat
from torch.utils.data import Dataset

import statistics

transform = None
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class ImageFolderDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(50, 50), split='train'):
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
        return ImageStat.Stat(image).mean, label
    
from torch.utils.data import random_split

dataset_dir = 'dataset/'
full_dataset = ImageFolderDataset(dataset_dir, transform=transform)

classes = full_dataset.classes

# Calculate the split sizes
num_samples = len(full_dataset)
train_size = int(num_samples * 1.0)
test_size = num_samples - train_size

# Split the dataset
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

import numpy as np
import random
from scipy.spatial.distance import squareform, pdist, cdist

def kmeans_fast(features, k, num_iters=1000):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find cdist (imported from scipy.spatial.distance) and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N = len(features)

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    features = np.array(features)

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = []
    for idx in idxs:
        centers.append(features[idx])
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        print(n)
        ### YOUR CODE HERE
        new_assignments = np.zeros(N, dtype=np.uint32)
        
        # Get new assignments for each feature vector
        new_assignments = np.argmin(cdist(features, centers), axis=1)
        
        # Break if none of the assignments changed
        if np.all(new_assignments == assignments):
            break
            
        # Calculate new cluster centers
        for i in range(k):
            centers[i] = np.mean(features[new_assignments == i], axis=0)
        print(assignments)
        assignments = new_assignments
        ### END YOUR CODE

    return assignments

train_dataset_features = [image[0] for image in train_dataset]
train_dataset_labels = [image[1] for image in train_dataset]
full_dataset_features = [image[0] for image in full_dataset]
full_dataset_labels = [image[1] for image in full_dataset]
assignments = kmeans_fast(full_dataset_features, len(classes))
print(assignments)
groups = {}
for i in range(len(classes)):
    groups[i] = []
for i in range(len(assignments)):
    groups[assignments[i]].append(full_dataset_labels[i])
group_modes = [statistics.mode(groups[i]) for i in range(len(classes))]
real_assignments = [group_modes[assignments[i]] for i in range(len(assignments))]
correct = 0
for i in range(len(real_assignments)):
    correct += (real_assignments[i] == full_dataset_labels[i])
print(correct)
print(correct / len(full_dataset_labels))