#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Ivan Gruber"
__version__ = "1.0.0"
__maintainer__ = "Ivan Gruber"
__email__ = "ivan.gruber@seznam.cz"

"""
Train script for nonBit models.
"""


import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import time
import copy
import torchvision.transforms as transforms
import sys

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def count_net_param(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


core_directory = ''

classes = ('outdoor', 'indoor')
num_classes = len(classes)
feature_extract = False
batch_size = 32
num_epochs = 20
dataset_size = 100000


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])


trainset = torchvision.datasets.ImageFolder(root=core_directory+'Data/',  transform=transform)
trainset, devset = torch.utils.data.random_split(trainset, [90000, 10000])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
devloader = torch.utils.data.DataLoader(devset, batch_size=batch_size, shuffle=True, num_workers=0)


writer = SummaryWriter(core_directory+'/runs/ResNet')

# rewrite according to trained architecture
net = models.resnet50(pretrained=True)
set_parameter_requires_grad(net, feature_extract)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, num_classes)
print(net)
print('Total number of parameters: '+str(count_net_param(net)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
print(device)

params_to_update = net.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in net.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in net.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
criterion = nn.CrossEntropyLoss()

best_model_wts = copy.deepcopy(net.state_dict())
best_acc = 0
best_epoch = 0
sys.stdout.flush()

for epoch in range(num_epochs):  # loop over the dataset multiple times
    start = time.time()
    net.train()
    train_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        train_loss += loss.item()
        if i % 1000 == 999:
            writer.add_scalar('training loss',
                              running_loss / 1000,
                              epoch * len(trainloader) + i)
            running_loss = 0.0

    net.eval()
    print('[%d] Train loss: %.3f' %
          (epoch + 1, train_loss / len(trainset)))

    test_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for j, data in enumerate(devloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('[%d]  Validation loss: %.3f' % (epoch + 1, test_loss / len(devset)))
    epoch_acc = 100 * correct / total
    print('Validation accuracy: %d %%' % (epoch_acc))
    scheduler.step(test_loss)
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_epoch = epoch + 1
        best_model_wts = copy.deepcopy(net.state_dict())
    end = time.time()
    print("Time for epoch: " + str(end - start))
    sys.stdout.flush()

print('Best validation accuracy: '+str(best_acc) + 'in epoch no.: '+str(best_epoch))
torch.save(best_model_wts, core_directory+'models/ResNet_best.pth')
torch.save(net.state_dict(), core_directory+'models/ResNet_last.pth')
print('Finished Training')

