# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:31:03 2023

@author: s.gorania
"""
# Import relevant libraries
import matplotlib.pyplot as plt
import torchvision
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# Download train and test sets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# Count the number of images
num_train_images = len(trainset)
num_test_images = len(testset)

print("Number of images in the training set:", num_train_images)
print("Number of images in the test set:", num_test_images)

# Show images
fig, ax = plt.subplots(4, 6, sharex=True, sharey=True, figsize=(12,8))
for i in range(0, 24):
    row, col = i//6, i%6
    ax[row][col].imshow(trainset.data[i])
plt.show()