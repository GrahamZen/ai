# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# import
from cifardef import *
from resnet import *
from PIL import Image  
import torch,math,os
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# %%
PATH='ResNet.pkl'
model = ResNet18()
if os.path.isfile(PATH):
    model.load_state_dict(torch.load(PATH))
    print(model.eval())
else:
    print('No model')
cuda_gpu = torch.cuda.is_available()
if cuda_gpu:
    model = model.cuda()


# %%
batch_size = 128
learning_rate = 0.1
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
test_dataset = cifar_10('data/cifar-10-batches-py', train=False, transform=transform_test)
meta=unpickle('data/cifar-10-batches-py/batches.meta')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_datasetList = [ cifar_10('data/cifar-10-batches-py', train=True, transform=transform_train,index=i+1) for i in range(5)]
train_loaderList=[DataLoader(t, batch_size=batch_size, shuffle=True) for t in train_datasetList]

# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


log=open(PATH.split('.')[0]+'.log','a')
best_acc=float(test(model, 1, criterion, test_loader)[0])


# %%
epochs1=135
epochs2=185
epochs3=240
epochs=epochs3
epoch=epochs3+1 # 不训练
# epoch=1 # 训练
optimizer = optim.SGD(model.parameters(), lr=0.1)
while epoch<=epochs1:
    print('Train Epoch: {}/{}: lr = {}'.format(epoch,epochs,0.1))
    print('Train Epoch: {}/{}:'.format(epoch,epochs),file=log)
    for t in train_loaderList:
        train(model,num_epoches,criterion,optimizer,t,file=log)
    if(epoch%(1)==0):
        acc, loss = test(model, 1, criterion, test_loader,log)
        if acc>best_acc:
            best_acc=acc
            print('better model,saved as',PATH)
            torch.save(model.state_dict(),PATH)
    epoch+=1
optimizer = optim.SGD(model.parameters(), lr=0.01)

while epoch<=epochs2:
    print('Train Epoch: {}/{}: lr = {}'.format(epoch,epochs,0.01))
    print('Train Epoch: {}/{}:'.format(epoch,epochs),file=log)
    for t in train_loaderList:
        train(model,num_epoches,criterion,optimizer,t,file=log)
    if(epoch%(1)==0):
        acc, loss = test(model, 1, criterion, test_loader,log)
        if acc>best_acc:
            best_acc=acc
            print('better model,saved as',PATH)
            torch.save(model.state_dict(),PATH)
    epoch+=1

optimizer = optim.SGD(model.parameters(), lr=0.001)

while epoch<=epochs3:
    print('Train Epoch: {}/{}: lr = {}'.format(epoch,epochs,0.001))
    print('Train Epoch: {}/{}:'.format(epoch,epochs),file=log)
    for t in train_loaderList:
        train(model,num_epoches,criterion,optimizer,t,file=log)
    if(epoch%(1)==0):
        acc, loss = test(model, 1, criterion, test_loader,log)
        if acc>best_acc:
            best_acc=acc
            print('better model,saved as',PATH)
            torch.save(model.state_dict(),PATH)  
    epoch+=1
log.close()

# %%
acc, loss = test(model, 1, criterion, test_loader)
test_visual(model,'data/cifar-10-batches-py',meta,transform_test)
