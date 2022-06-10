from PIL import Image
import os
import os.path
import copy
import errno
import numpy as np
import sys
import pickle
import random
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

import torch
import torch.nn.functional as F
from torch.autograd import Variable as V
import torchvision.transforms as transforms


class SavedCIFAR(data.Dataset):

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    # normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.307, 122.961, 113.8575]],
    #                                  std=[x / 255.0 for x in [51.5865, 50.847, 51.255]])

    augment_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                            (4, 4, 4, 4), mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    easy_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    def __init__(self, root='', train_file='', label_file='', true_label_file='', train = True, transform='easy', target_transform=None):
        self.root = root
        self.data = np.load(os.path.join(root, train_file))
        self.labels = np.load(os.path.join(root, label_file))
        self.true_labels = np.load(os.path.join(root, true_label_file))
        if transform == 'easy':
            self.transform = self.easy_transform
        elif transform == 'hard':
            self.transform = self.augment_transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        img, target, true_target = self.data[index], self.labels[index], self.true_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            true_target = self.target_transform(true_target)

        return img, target, index, true_target

    def __len__(self):
        return len(self.data)
