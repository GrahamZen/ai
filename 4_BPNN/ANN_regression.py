#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import random as rd
import math
import copy


# In[ ]:


def sigmoid(x):
    mask = (x > 0)
    positive_out = np.zeros_like(x, dtype='float64')
    negative_out = np.zeros_like(x, dtype='float64')
    
    # 大于0的情况
    positive_out = 1 / (1 + np.exp(-x, positive_out, where=mask))
    # 清除对小于等于0元素的影响
    positive_out[~mask] = 0
    
    # 小于等于0的情况
    expX = np.exp(x,negative_out,where=~mask)
    negative_out = expX / (1+expX)
    # 清除对大于0元素的影响
    negative_out[mask] = 0
    
    return positive_out + negative_out        
def invSigmoid(x):
    return [math.log(i/(1-i))for i in x]
def sigmoid_prime(sigmoidx):
    return np.multiply(sigmoidx,1-sigmoidx)


# In[ ]:


class ANN:
    def __init__(self,sizes):
#         输入层之外的层数,例子为1+2，layNum=3
        self.num_layers =len(sizes)
        self.sizes=sizes
#         i索引W(i+1)矩阵
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
    def forward_pass(self,x):
        a=np.matrix(x).transpose()
        for w,b in zip(self.weights,self.biases):
            z=w*a+b
            a=sigmoid(z)
        return float(z)
    
    def backPropagation(self,x,y):
        grad_w=[np.zeros(w.shape)for w in self.weights]
        grad_b=[np.zeros(b.shape)for b in self.biases]
        delta=[np.zeros(b.shape)for b in self.biases]
        a=np.matrix(x).transpose()
        alist=[a]
        zlist=[]
        for w,b in zip(self.weights,self.biases):
            z=w*a+b
            a=sigmoid(z)
            zlist.append(z)
            alist.append(a)
        alist[-1]=zlist[-1]
        delta[-1]=(z-y).transpose()
        for l in range(2,self.num_layers):
            delta[-l]=np.multiply(sigmoid_prime(alist[-l]),self.weights[1-l].transpose()*delta[1-l])
        for i in range(self.num_layers-1):
            grad_w[i]=delta[i]*np.mat(alist[i]).transpose()
        return grad_w, delta
    def SGD(self,train_set,validation_set,batch_size,lr=1e-3,epochs=1e3):
        variate=float('inf')
        besta=None
        for j in range(int(epochs)):
            np.random.shuffle(train_set)
            mini_batches = [train_set[k:k+batch_size] for k in range(0, train_set.shape[0], batch_size)]
            for mini_batch in mini_batches:
                self.minibatch(mini_batch, eta)
            curVar=self.validate(validation_set)
            if curVar<variate:
                variate=curVar
                besta=copy.deepcopy(self)
            print("Epoch {} : {}, best : {}".format(j,curVar,variate));
        return besta
    def gradient_descent(self,x,y,lr=1e-3):
        grad_w,grad_b=self.backPropagation(x,y)
        self.weights=[w-lr*gw for w,gw in zip(self.weights,grad_w)]
        self.biases=[b-lr*gb for b,gb in zip(self.biases,grad_b)]
    def train(self,train_set,lr=1):
        for row in train_set:
            self.gradient_descent(row[:-1],row[-1],lr)
    def minibatch(self,batch,lr=1e-3):
        batchsize=len(batch)
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for row in batch:
            delta_gw,delta_gb=self.backPropagation(row[:-1],row[-1])
            grad_w=[gw+dgw for gw,dgw in zip(grad_w,delta_gw)]
            grad_b=[gb+dgb for gb,dgb in zip(grad_b,delta_gb)]
        self.weights=[w-lr/batchsize*gw for w,gw in zip(self.weights,grad_w)]
        self.biases=[b-lr/batchsize*gb for b,gb in zip(self.biases,grad_b)]
    def validate(self,validation_set):
        arr=[self.forward_pass(row[:-1])for row in validation_set]
        return sum(np.square(arr-validation_set[:,-1]))/validation_set.shape[0]
def preprocess(dataSet):
    dataSet.dteday=dataSet.dteday.map(date2int)
    maxdte=float(max(dataSet.dteday))
    dataSet.dteday=dataSet.dteday.map(lambda x:float(x)/maxdte)
    return dataSet.values[:,1:]
def random_choice(train_set,batchSize):
    return train_set[np.random.randint(0,train_set.shape[0],10), :]
def date2int(str):
    l=[int(x) for x in str.split('/')]
    return l[0]+l[1]*30+l[2]
def calcVar(ds,a):
    arr=[]
    for i in range(ds.shape[0]):
        arr.append(a.forward_pass(ds[i][:-1]))
    return np.var((np.matrix(arr)-ds[:,-1]))
def split(dataSet,proportion):
    np.random.shuffle(dataSet)
    train_num=int(dataSet.shape[0]*proportion)
    return dataSet[0:train_num,:],dataSet[train_num:,:]
def compute_eta_t(eta_min, eta_max, T_cur, Ti):
    '''Equation (5).
    # Arguments
        eta_min,eta_max,T_cur,Ti are same as equation.
    # Returns
        eta_t
    '''
    pi = np.pi
    eta_t = eta_min + 0.5 * (eta_max - eta_min) * (np.cos(pi * T_cur / Ti) + 1)
    return eta_t


# In[ ]:


# 初始化
dataSet=pd.read_csv('lab4_dataset/train.csv')
ds=preprocess(dataSet)
ts,vs=split(ds,0.8)
inputNode=ds.shape[1]-1


# In[ ]:


# 双隐藏层
learningRate=0.1
maxj=5
for j in range(maxj):
    a=ANN([inputNode,80,80,1])
    eta=learningRate/(10**j)
    besta = a.SGD(ts,vs,128,eta,500)
    deviate = besta.validate(vs)
    print('d:',deviate,'lr = ',eta )
#         if((deviate-min_deviate)/min_deviate>0.01 or abs(deviate-min_deviate)<1e-2):


# In[ ]:


# 单隐藏层
learningRate=0.1
maxj=5
for j in range(maxj):
    a=ANN([inputNode,80,1])
    eta=learningRate/(10**j)
    besta = a.SGD(ts,vs,128,eta,500)
    deviate = besta.validate(vs)
    print('d:',deviate,'lr = ',eta )
#         if((deviate-min_deviate)/min_deviate>0.01 or abs(deviate-min_deviate)<1e-2):


# In[ ]:


# 节点个数
learningRate=0.0001
maxj=5
for j in range(maxj):
    a=ANN([inputNode,int(256/(2**j)),1])
    eta=learningRate
    besta = a.SGD(ts,vs,128,eta,500)
    deviate = besta.validate(vs)
    print('d:',deviate,'lr = ',eta )
#         if((deviate-min_deviate)/min_deviate>0.01 or abs(deviate-min_deviate)<1e-2):


# In[ ]:


# mini-batch size
learningRate=0.001
maxj=5
for j in range(maxj):
    a=ANN([inputNode,256,1])
    eta=learningRate
    besta = a.SGD(ts,vs,int(1024/(2**j)),eta,500)
    deviate = besta.validate(vs)
    print('d:',deviate,'lr = ',eta )
#         if((deviate-min_deviate)/min_deviate>0.01 or abs(deviate-min_deviate)<1e-2):


# In[ ]:


# 单隐藏层
learningRate=0.1
maxj=5
for j in range(maxj):
    a=ANN([inputNode,80,80,80,1])
    eta=learningRate/(10**j)
    besta = a.SGD(ts,vs,128,eta,500)
    deviate = besta.validate(vs)
    print('d:',deviate,'lr = ',eta )
#         if((deviate-min_deviate)/min_deviate>0.01 or abs(deviate-min_deviate)<1e-2):


# In[ ]:


# 最好训练
learningRate=1e-4
a=ANN([inputNode,80,80,1])
besta=a
maxj=10
for j in range(maxj):
    a=besta
    eta=compute_eta_t(0,learningRate,j,maxj)
    besta = a.SGD(ts,vs,128,eta,1000)
    deviate = besta.validate(vs)
    print('d:',deviate,'lr = ',eta )
#         if((deviate-min_deviate)/min_deviate>0.01 or abs(deviate-min_deviate)<1e-2):

