#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Ivan Gruber"
__version__ = "1.0.0"
__maintainer__ = "Ivan Gruber"
__email__ = "ivan.gruber@seznam.cz"

"""
Test script for nonBit models.
"""

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import time
import torchvision.transforms as transforms
import copy
import h5py
import numpy as np
import time

classes = ('outdoor', 'indoor')
num_classes = len(classes)
batch_size = 8

def count_net_param(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


# rewrite according to tested architecture

net = models.vgg16_bn(pretrained=False)
num_ftrs = net.classifier[6].in_features
net.classifier[6] = nn.Linear(num_ftrs,2)

# net = models.densenet161(pretrained=False)
# num_ftrs = net.classifier.in_features
# net.classifier = nn.Linear(num_ftrs, num_classes)

# net = models.resnet50(pretrained=False)
# num_ftrs = net.fc.in_features
# net.fc = nn.Linear(num_ftrs, num_classes)


net.load_state_dict(torch.load('models/VGG16_best.pth'))
net.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
# print(net)
print('Total number of parameters: '+str(count_net_param(net)))

# load test data from h5
with h5py.File('./h5py/test.h5','r') as fr:
    imgs = fr['data'][:1000,0, :,:,:]
    labels = fr['label'][:1000,0]


mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
imgs = imgs/255.
imgs = (imgs - mean)/std
imgs = np.transpose(imgs,(0,3,1,2))

img_tensor = torch.Tensor(imgs)

label_tensor = torch.Tensor(labels)
label_tensor = label_tensor.type(torch.int64)

test_set = torch.utils.data.TensorDataset(img_tensor, label_tensor)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)


correct = 0
total = 0
total_time = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        start = time.time()
        outputs = net(images)
        stop = time.time()
        total_time += stop - start
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 1000 test images: %5.2f %%' % (
    100 * correct / total))
print(total_time)
