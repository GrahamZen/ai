#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt
import random
import pandas as pd
import copy


# In[2]:


def PLA(train_set,eta=1, maxIter=1e2):
    w=np.zeros((train_set.shape[1]-1))
    learnRate=1.0
    iterTime=0
    bestacc=0.0
    bestw=w
    while iterTime<maxIter:
        miss=0
        accuracy=0.0
        for row in train_set:
            x=row[:-1]
            res=np.sign(np.dot(x,w))
            if res==row[-1]:
                pass
            else:
                miss+=1
                w+=eta * row[-1] * x
        if miss==0:
            break
        iterTime+=1
        accuracy=(train_set.shape[0]-miss)/train_set.shape[0]
        if(accuracy>bestacc):
            bestacc=accuracy
            bestw=w
#             print('accuracy=',accuracy)
        res = test(train_set[:,:-1], w)
        print(iterTime,":")
        for i in res:
            print(int(i))
        print("------------------------------")
    return bestw


# In[3]:


def k_fold(dataSet,k,i):
    avg_size=dataSet.shape[0]/k
    begin=int(avg_size*(i))
    end=int(avg_size*(i+1))
    train_set=np.delete(dataSet,np.s_[begin:end],axis=0)
    return train_set, dataSet[begin:end]


# In[4]:


def PLA_pocket(train_set,eta=1, maxIter=1e2):
    w=np.zeros((train_set.shape[1]-1))
    learnRate=1.0
    iterTime=1
    bestacc=0.0
    bestw=w
    accuracy=0.0
    while iterTime<maxIter:
#         learnRate=eta*1.0/math.sqrt(iterTime)
        learnRate=eta
        x=random.choice(train_set)
        xres=x[-1]
        x=x[:-1]
        res=np.sign(np.dot(x,w))
        if res==xres:
            pass
        else:
            w+=x * xres *learnRate
            accuracy=validate(train_set,w)
            if(accuracy>bestacc):
#                 print('accuracy=',bestacc)
                bestacc=accuracy
                bestw=w
        iterTime+=1
        if accuracy==1:
            return bestw
    return bestw


# In[5]:


def test(testSet,w):
    return np.sign(testSet*np.mat(w).transpose())
def validate(dataSet,w):
    res=test(dataSet[:,:-1],w)
    return 1-np.mean(abs(res-np.mat(dataSet[:,-1]).transpose()))/2


# In[11]:


def evaluate(filename, maxIter=1e3, method=PLA, eta=1,k=10):
    file=np.loadtxt(filename,dtype=float,delimiter=',')
    file[:,-1][file[:,-1]==0]=-1
    dataSet=np.ones((file.shape[0],file.shape[1]+1))
    dataSet[:,:-2]=file[:,:-1]
    dataSet[:,-1]=file[:,-1]
    accuracy=0.0
    for i in range(k):
        s1,s2=k_fold(dataSet,k,i)
        w=method(s1,eta,maxIter)
        currAc=validate(s2,w)
        accuracy+=currAc
        print(i," finished, accuracy:",currAc)
    return accuracy/k


# In[7]:


def main():
    file=np.loadtxt('test.csv',dtype=float,delimiter=',')
    file[:,-1][file[:,-1]==0]=-1
    train_set=np.ones((file.shape[0],file.shape[1]+1))
    train_set[:,:-2]=file[:,:-1]
    train_set[:,-1]=file[:,-1]
    w=PLA(train_set,1,10)
    res=test(train_set[:,:-1],w)
    for i in res:
        print(int(i))

# In[8]:


main()
