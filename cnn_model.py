from google.colab import drive
drive.mount('/content/drive')

import torch
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from  torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
from skimage import io, transform


import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import csv
import os
import math
import cv2

train_data = torchvision.datasets.ImageFolder('drive/My Drive/dissertation/train/4cate/',transform=transforms.Compose([transforms.Resize((224,224)),transforms.CenterCrop(224),transforms.ToTensor()]))
test_data = torchvision.datasets.ImageFolder('drive/My Drive/dissertation/test/4cate/',transform=transforms.Compose([transforms.Resize((224,224)),transforms.CenterCrop(224),transforms.ToTensor()]))
valid_data = torchvision.datasets.ImageFolder('drive/My Drive/dissertation/valid/4cate/',transform=transforms.Compose([transforms.Resize((224,224)),transforms.CenterCrop(224),transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64,shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64,shuffle=True)

class ConvNet(nn.Module):
    
    def __init__(self, num_classes=4):
     # Add network layers here
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(512*12*12,4096)

        self.fc2 = nn.Linear(4096,5)

        #self.final = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.3)
    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)

        output = self.conv3(output)
        output = self.relu(output)

        output = self.conv4(output)
        output = self.relu(output)
        output = self.pool(output)

        output = self.conv5(output)
        output = self.relu(output)

        output = self.conv6(output)
        output = self.relu(output)
        output = self.pool(output)

        output = self.conv7(output)
        output = self.relu(output)

        output = self.conv8(output)
        output = self.relu(output)
        output = self.pool(output)

        #print(output.shape)

        output = output.reshape(output.size(0),-1)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)

        output = self.fc2(output)
                
        #output = self.final(output)

        return output
model = ConvNet()

model.fc1.weight

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

model_gpu = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_gpu.parameters(), lr=0.001, momentum=0.9)

import timeit
train_losses, valid_losses = [],[]
def train_model_epochs(num_epochs):    
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    running_loss = 0.0
    

    model.train()
    for i, data in enumerate(train_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)            
            optimizer.zero_grad()          
            outputs = model_gpu(images)
            _, preds = torch.max(outputs, 1)
            
            loss = criterion(outputs, labels)          
            loss.backward()

            optimizer.step()            
            running_loss += loss.item()
            if i % 50 == 49:    
                print('Epoch / Batch [%d / %d] - Loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    else:
       valid_loss = 0
       accuracy = 0
       test_accuracy = 0
       with torch.no_grad():
          model.eval()
          for i, data in enumerate(valid_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device) 
            outputs = model_gpu(images)
            
            loss = criterion(outputs, labels)
            valid_loss += loss

            otpt = torch.exp(outputs)
            top_p, top_class = otpt.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

          for i, data in enumerate(test_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device) 
            outputs = model_gpu(images)

            otpt = torch.exp(outputs)
            top_p, top_class = otpt.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor))

       train_losses.append(running_loss/len(train_loader))
       valid_losses.append(valid_loss/len(valid_loader))

       print("Epoch: {}/{}..".format(epoch+1, epoch),
           "Training Loss: {:.3f}..".format(running_loss/len(train_loader)),
           "Valid Loss: {:.3f}..".format(valid_loss/len(valid_loader)),
           "Valid Accuracy: {:.3f}..".format(accuracy/len(valid_loader)),
           "Test Accuracy: {:.3f}..".format(test_accuracy/len(test_loader))
        )

gpu_train_time = timeit.timeit(
    "train_model_epochs(num_epochs)",
    setup="num_epochs=50",
    number=1,
    globals=globals(),
)

correct = 0
total = 0
label_list = []
pre_list =[]

with torch.no_grad():
    
    # Iterate over the test set
    for data in valid_loader:
        images, labels = data
        
        images = images.to(device)
        labels = labels.to(device)
        
        
        outputs = model_gpu(images)
        
        _, predicted = torch.max(outputs.data, 1)
        label_list += labels.tolist()
        pre_list += predicted.tolist()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(label_list, pre_list)
cm

gpu_train_time