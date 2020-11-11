from PIL import Image
import torch
import math
import os
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from matplotlib import pyplot as plt
import numpy as np
import sys
# loader and tester definition

class cifar_10(torch.utils.data.Dataset):
    def __init__(self, root, train=False, transform=None, target_transform=None, index=1):
        if train == True:
            data = unpickle(root+'/data_batch_'+str(index))
        else:
            data = unpickle(root+'/test_batch')
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = array2img(self.data[b'data'][index])
        label = self.data[b'labels'][index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data[b'data'])


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_channel(data, channel='r'):
    if channel == 'r':
        return data[0:1024].reshape(32, 32)
    if channel == 'g':
        return data[1024:2048].reshape(32, 32)
    if channel == 'b':
        return data[2048:3072].reshape(32, 32)


def array2img(data):
    imgArr = np.array([get_channel(data, s)
                       for s in ['r', 'g', 'b']]).transpose(1, 2, 0)
    image = Image.fromarray(imgArr)
#     plt.imshow(image)
    return image


def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img


def show_from_tensor(tensor, title=None):
    img = tensor.clone()
    img = tensor_to_np(img)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def test_visual(model, test_dataset_file, meta_data, transform=None):
    model.eval()
    test_loader = DataLoader(
        cifar_10(test_dataset_file, train=False, transform=transform), shuffle=False)
    test_loader_origin = DataLoader(cifar_10(
        test_dataset_file, train=False, transform=transforms.ToTensor()), shuffle=False)
    if torch.cuda.is_available():
        model = model.cuda()
    for data, origin_data in zip(test_loader, test_loader_origin):
        img, label = data
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        with torch.no_grad():
            out = model(img)
        _, pred = torch.max(out, 1)
        show_from_tensor(origin_data[0])
        print(meta_data[b'label_names'][pred], ',',
              meta_data[b'label_names'][label])
# train/test method


def train(model, epoch, criterion, optimizer, data_loader, file=sys.stdout):
    cuda_gpu = torch.cuda.is_available()
    model.train()
    total = 0
    all = len(data_loader.dataset)
    for batch_idx, (data, target) in enumerate(data_loader):
        if cuda_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total += data.shape[0]
        if total % (all/2) == 0:
            if file != sys.stdout:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, total, all, 100. * total / all, loss.data), file=file)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, total, all, 100. * total / all, loss.data))


def test(model, epoch, criterion, data_loader, file=sys.stdout):
    cuda_gpu = torch.cuda.is_available()
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        if cuda_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        with torch.no_grad():
            output = model(data)
        test_loss += criterion(output, target).data
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    # loss function already averages over batch size
    test_loss /= len(data_loader)
    acc = torch.true_divide(correct, len(data_loader.dataset))
    if file != sys.stdout:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(data_loader.dataset), 100. * acc), file=file)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(data_loader.dataset), 100. * acc))
    return (acc, test_loss)
