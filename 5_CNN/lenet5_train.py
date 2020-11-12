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


# %%
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


# %%
batch_size = 128
learning_rate = 0.0001
num_epoches = 1000

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
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
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）


# %%
epoch=1


# %%
epochs=0

while epoch<=epochs:
    print('Train Epoch: {}/{}:'.format(epoch,epochs))
    for t in train_loaderList:
        train(model,num_epoches,criterion,optimizer,t,log)
    if(epoch%(1)==0):
        acc, loss = test(model, 1, criterion, test_loader,log)
    epoch+=1

torch.save(model.state_dict(),PATH)
log.close()


# %%
test(model, 1, nn.CrossEntropyLoss(), test_loader)
test_visual(model,'data/cifar-10-batches-py',meta,transform_test)

