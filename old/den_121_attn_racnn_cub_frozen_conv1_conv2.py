# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import math
import sys
from tqdm import tqdm
sys.path.append('densenet_model/')
from densenet import densenet121_attn_racnn
import matplotlib as plt
plt.use('Agg')

start = time.time()
exp_name = sys.argv[0].split('.')[0]
IMG_H, IMG_W = 299, 299

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# for cropped data_dir = '../CUB_200_2011/cropped_test'
data_dir = '../CUB_200_2011/cub_299_299'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'validation']}
dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'validation']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

# Get a batch of training data
inputs, classes = next(iter(dataloders['train']))

def save_plot(imgs):
    """
    Takes a (299, 299, 3)
    Glimpses: (3, 3, 299, 299)
    """
    plt.figure(figsize=(10,10))
    n = len(imgs)
    x = math.ceil(n/2)
    y = 2

    for i, img in enumerate(imgs):
        plt.subplot(x,y,i+1)
        plt.axis('off')
        plt.imshow(img)
    plt.savefig(name + '.png')

def save_glimpses(x, y, glimpses, epoch):
    """
    glimpses: (s, g, 3, 299, 299)
    """
    glimpses = glimpses.permute(0, 1, 3, 4, 2)
    glimpses = glimpses.data.numpy()
    x = x.permute(0, 2, 3, 1)
    for s in glimpses.shape[0]:
        name = "{}_{}_{}".format(exp_name, class_names[y], epoch)
        img = x[s].numpy()
        save_plot(img, glimpses[s], name)

def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    best_acc = 0.0
    for epoch in range(num_epochs):
        start_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in tqdm(dataloders[phase]):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs, glimpses = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(),
                        'models/{}_{}_{}.pth'.format(exp_name, epoch, best_acc))

        time_elapsed = time.time() - start_time
        print('Time: {}'.format(time_elapsed))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model


print('Loading model')
# Load a pretrained model and reset final fully connected layer.
model_ft = densenet121_attn_racnn(freeze_conv1=True, freeze_conv2=True) 

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(filter(lambda p:p.requires_grad, model_ft.parameters()), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

end = time.time()
print("Took {}s".format(end-start))
print('Training')
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=60)
