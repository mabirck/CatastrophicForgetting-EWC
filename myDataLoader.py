import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms, datasets
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt


def getDataLoader(origin, train, download=True, permutation=None, args=None):

    transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.ToPILImage(),
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,)),
           transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation))

    ])

    dataset = datasets.MNIST(
        './datasets/{name}'.format(name=origin), train=train,
        download=download, transform=transform,)

    trainloader = torch.utils.data.DataLoader(dataset,
    batch_size=args.batch_size if train else args.test_batch_size,
    shuffle=True, num_workers=2)

    return trainloader;


def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    image.view(c, h, w)
    return image
