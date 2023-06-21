# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 12:13:04 2023

@author: s.gorania
"""
import matplotlib.pyplot as plt
import torchvision
import torch
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.optim as optim
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""
In the first code block, we access the original NumPy arrays,
while in the second code block, we access and display the transformed
tensor images. This leads to a difference in the displayed images.
"""
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
trainset = CIFAR10(root='./data', train=True, download=True,
                   transform=transform)
testset = CIFAR10(root='./data', train=False, download=True,
                  transform=transform)

batch_size = 24
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=True)

fig, ax = plt.subplots(4, 6, sharex=True, sharey=True, figsize=(12, 8))
for images, labels in trainloader:
    for i in range(batch_size):
        row, col = i//6, i % 6
        ax[row][col].imshow(images[i].numpy().transpose([1, 2, 0]))
    break  # take only the first batch
#plt.show()

# Specify 3 colour channels (RGB) with 32x32 pixels
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=1),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2)),
    nn.Flatten(),
    nn.Linear(8192, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    for inputs, labels in trainloader:
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            y_pred = model(inputs)
            acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)
    acc /= count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))
