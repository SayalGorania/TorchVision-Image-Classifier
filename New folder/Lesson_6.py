# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 12:00:39 2023

@author: s.gorania
"""

import matplotlib.pyplot as plt
import torchvision
import torch
from torchvision.datasets import CIFAR10

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
plt.show()
