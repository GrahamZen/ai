#!/usr/bin/env python
# coding: utf-8

# In[125]:


import numpy as np
import math
import matplotlib.pyplot as plt
import random
import pandas as pd
import copy


# In[195]:


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
def gradient(dataSet, w):
    grad = 0.0
    x = dataSet[:, :-1]
    w = np.mat(w)
    label=np.mat(dataSet[:, -1]).transpose()
    return x.transpose() * (label - sigmoid(x * w))

def gradientDescent(dataSet, iterTime=1e2, learningRate=1e-5):
    i = 0
    w = np.zeros(((dataSet.shape[1] - 1), 1))
    while i < iterTime:
        gd = gradient(dataSet, w)
        w += learningRate * gd
        i += 1
        print(i,":")
        res=test(dataSet[:,:-1],w)
        for jj in res:
            print(int(jj))
        print("--------------")
    return w

def gradientDescentLog(dataSet, iterTime, learningRate):
    log={}
    i = 0
    w = np.zeros(((dataSet.shape[1] - 1), 1))
    while i < iterTime:
        gd = gradient(dataSet, w)
#         print((np.linalg.norm(gd)))
        w += learningRate * gd
        i += 1
        log[i]=copy.deepcopy(w)
    return w,log


# In[178]:


def test(testSet,w):
    res=sigmoid(testSet*np.mat(w))
    res[res>0.5]=1
    res[res<0.5]=0
    return res
def validate(dataSet,w):
    res=test(dataSet[:,:-1],w)
    return 1-np.mean(abs(res-np.mat(dataSet[:,-1]).transpose()))


# In[169]:


def k_fold(dataSet,k,i):
    avg_size=dataSet.shape[0]/k
    begin=int(avg_size*(i))
    end=int(avg_size*(i+1))
    train_set=np.delete(dataSet,np.s_[begin:end],axis=0)
    return train_set, dataSet[begin:end]
def evaluate(dataSet, maxIter=1e3, eta=1, k=10):
    accuracy=0.0
    for i in range(k):
        s1,s2=k_fold(dataSet,k,i)
        w=gradientDescent(s1,maxIter,eta)
        currAc=validate(s2,w)
        accuracy+=currAc
#         print(i," finished, accuracy:",currAc)
    return accuracy/k
def train(dataSet,testSet, maxIter=1e3, eta=1):
    w=gradientDescent(dataSet,maxIter,eta)
    


# In[166]:


def train_all(dataSet, maxIter=1e3, eta=1, k=10):
    accuracy=0.0
    accuracy_dict={}
    for j in range(maxIter):
        accuracy_dict[j+1]=0.0
    for i in range(k):
        s1,s2=k_fold(dataSet,k,i)
        w,log=gradientDescentLog(s1,maxIter,eta)
        for j in log:
            accuracy_dict[j]+=validate(s2,log[j])
    for j in range(maxIter):
        accuracy_dict[j+1]/=k
    return accuracy/k,accuracy_dict


# In[170]:


def train_toTable(maxIter=5000,eta=1):
    d={}
    k=maxIter
    for i in range(k):
        d[i+1]={}
    for j in range(20):
        eta/=math.pow(10,1/2.5)
        res,log=train_all(train_set,k,eta)
        for i in log:
            d[i][eta]=copy.deepcopy(log[i])
        print('T:',k,', eta:',eta,':',res)
    pd.DataFrame(d).transpose().to_csv('res.csv')


# In[196]:


def main():
    file=np.loadtxt("test.csv",dtype=float,delimiter=',')
    train_set=np.ones((file.shape[0],file.shape[1]+1))
    train_set[:,:-2]=file[:,:-1]
    train_set[:,-1]=file[:,-1]
    w=gradientDescent(train_set,10,1)
    res=test(train_set[:,:-1],w)
    for i in res:
        print(int(i))
    


# In[197]:


main()

