#CNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import DatasetFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets.folder import default_loader
import numpy as np
import os
import cv2
from torch.utils.data import TensorDataset
from torchsummary import summary
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

#loading and preprocessing
dataset_folder = "GTSRB_subset_2"
device = "cuda"

# Defining transformer to convert image data to tensors
transform = transforms.Compose([
        transforms.ToTensor()   
])

# Generate all valid data in the form of tensors from the given  root directory
all_data = DatasetFolder(
    root=dataset_folder, 
    loader=default_loader, 
    transform=transform, 
    extensions=(".png", ".jpg", ".jpeg")
               )

features = [features[0] for features in all_data]
labels = [labels[1] for labels in all_data]

#stratified split between classes
stratified_split = StratifiedShuffleSplit(test_size=0.2,n_splits=1, random_state=42)
train_indices, test_indices = next(stratified_split.split(features, labels))

# extract train and test data seperately
train_data = torch.utils.data.Subset(all_data, train_indices)

# Extract test images and test labels
test_images = torch.utils.data.Subset(features, test_indices)
test_labels = torch.utils.data.Subset(labels, test_indices)

# Convert subset to tensor
test_images_tensor = torch.stack([image for image in test_images])

# Generate data loader 
train_loader = DataLoader(train_data, batch_size=32)

print("Train Dataset shape:", len(train_data))
print("Test Dataset shape:", len(test_images))

#model buliding
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      #con2d(inchannel,outchannel,kernel/filter/window size/stride 
      self.conv1 = nn.Conv2d(3, 10 , 3, 2)
      self.maxpool1 = nn.MaxPool2d(2)
      self.conv2 = nn.Conv2d(10, 10, 3, 2)
      self.maxpool2 = nn.MaxPool2d(2)
      self.fc = nn.Linear(90, 2)

    # x represents our data
    def forward(self, x):
      x = self.conv1(x)
      x = F.relu(x)
      x = self.maxpool1(x) 
      x = self.conv2(x)
      x = F.relu(x)
      x = self.maxpool2(x) 
      x = torch.flatten(x, 1,-1)
      x = self.fc(x)
      return x


modelCNN = Net().to(device)
print(modelCNN)

summary(modelCNN,input_size=(3,64,64), device = device)
num_epochs = 20
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(modelCNN.parameters(),lr=0.1)

# Training loop
print('number of epochs:',num_epochs)
print('Loss after each epoch')
for n in range(num_epochs):
    loss_e =0;
    for tr_image, tr_label in train_loader:
        y_pred = modelCNN(tr_image.to(device))
        loss = loss_fn(y_pred.float(), tr_label.to(device))
        loss_e += loss.item()*tr_image.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss_e / len(train_loader.sampler))


test_labels_onehot = modelCNN(test_images_tensor.float().to(device)) 
test_labels_p = np.argmax(test_labels_onehot.detach().cpu().numpy(),axis=1)
test_acc = 1-(np.count_nonzero(test_labels-test_labels_p)/len(test_labels))

print('\nTest accuracy:', test_acc)
