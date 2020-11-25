# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from cifardef import *
from lenet5 import *
from PIL import Image  
import torch,math,os
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# %%加载已训练好的模型
PATH='Lenet.pkl'
model = lenet5(3,10)
if os.path.isfile(PATH):
    model.load_state_dict(torch.load(PATH))
    print(model.eval())
else:
    print('No model')
log=open(PATH.split('.')[0]+'.log','a')
best_acc=0
cuda_gpu = torch.cuda.is_available()
if cuda_gpu:
    model = model.cuda()


# %%处理数据集，定义训练超参数等
batch_size = 128
learning_rate = 0.0001
num_epoches = 1000

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_dataset = cifar_10('data/cifar-10-batches-py',
                        train=False, transform=transform_test)
meta=unpickle('data/cifar-10-batches-py/batches.meta')

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_datasetList = [ cifar_10('data/cifar-10-batches-py',
                         train=True, transform=transform_train,index=i+1) for i in range(5)]
train_loaderList=[DataLoader(t, batch_size=batch_size, shuffle=True) for t in train_datasetList]

# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)



