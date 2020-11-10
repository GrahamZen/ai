#!/usr/bin/env python
import sys
import pandas as pd
from math import log
from collections import Counter
log2=lambda x:log(x)/log(2)

def rel_entropy(dataSet,feat):
    currfeat=dataSet.loc[:,[feat,'Label']]
    featCount=Counter(currfeat.iloc[:,0])
    part_ent=0.0
    ent=0.0
    for feature in featCount:
        proportion=featCount[feature]/dataSet.shape[0]
        LabelCnt=Counter(currfeat.loc[currfeat[feat]==feature]['Label'])
        totalCnt=sum(LabelCnt.values())
        for r in LabelCnt.keys():
            p=float(LabelCnt[r])/totalCnt
            part_ent=part_ent-p*log2(p)
        ent+=part_ent*proportion
        part_ent=0
    return ent
def entropy(dataSet,feature='Label'):
    ent=0.0
    exentCnt=Counter(dataSet.loc[:,feature])
    for res in exentCnt:
        p=float(exentCnt[res])/dataSet.shape[0]
        ent-=p*log2(p)
    return ent
def getBestSplit(dataSet,feat):
    minGiniIndex=1.0
    BestSplitVal=None
    featLst=list(Counter(dataSet[feat]))
    for featVal in featLst:
        currfeat=dataSet.loc[:,[feat,'Label']]
        featCount=Counter(currfeat.iloc[:,0])
        oneSubtractGini=0.0
        giniIndex=0.0
        target_group=currfeat.loc[currfeat[feat]==featVal]['Label']
        another_group=currfeat.loc[currfeat[feat]!=featVal]['Label']
        portion=target_group.size/currfeat.shape[0]
        targetCnt=Counter(target_group)
        anotherCnt=Counter(another_group)
        p1=float(targetCnt[0])/target_group.size
        p2=float(anotherCnt[0])/another_group.size
        giniIndex+=2*(1-p1)*p1*portion+(2*(1-p2)*p2)*(1-portion)
        if(giniIndex<minGiniIndex):
            BestSplitVal=featVal
            minGiniIndex=giniIndex
    return BestSplitVal,minGiniIndex
def getBestFeat_gini(dataSet):
    minGiniIndex=1
    BestFeat=None
    BestFeatVal=None
    giniIndex=1
    currfeatVal=None
    for feat in dataSet.columns[:-1]:
        featLst=list(Counter(dataSet[feat]))
        if(len(featLst)==1):
            pass
        else:
            currfeatVal,giniIndex=getBestSplit(dataSet,feat)
            if giniIndex<minGiniIndex:
                BestFeat=feat
                BestFeatVal=currfeatVal
                minGiniIndex=giniIndex
    return BestFeat,BestFeatVal
def getBestFeature(dataSet,method='C4.5'):
    maxEntGain=0.0
    BestFeat=None
    entGain=0.0
    exEnt=entropy(dataSet)
    for feat in dataSet.columns[:-1]:
        entGain=exEnt-rel_entropy(dataSet,feat)
        if method=='ID3':
            pass
        elif method=='C4.5':
            entGain/=entropy(dataSet,feat)
        if(maxEntGain<entGain):            
            maxEntGain=entGain
            BestFeat=feat
    return BestFeat
class decisionnode:
    def __init__(self,child,feature=None,result=None,method=None,CART_val=None):
        self.feature=feature
        self.result=result
        self.child=child
        self.CART_val=CART_val
        self.method=method
def CreateTree(dataSet,featDict,method='C4.5',parent_res=-1):
    child={}
# 数据集为空，等于上一步没有分
    if dataSet.empty:
        return decisionnode(result=parent_res,feature=None,child=None)
    resList=dataSet.loc[:,'Label']
    labelCnt=Counter(resList)
    parent_res=max(labelCnt, key=labelCnt.get)
# 只有一种结果，已完全分开
    if len(labelCnt)==1:
        return decisionnode(result=resList.values[0],feature=None,child=None)
# 没有可以分的特征了
    if (dataSet.shape[1]==1) and (method!= 'CART'):
        return decisionnode(result=parent_res,feature=None,child=None)
    bestVal=None
    if method=='CART':
        bestfeat,bestVal=getBestFeat_gini(dataSet)
        if(bestfeat==None):
            return decisionnode(result=0,feature=None,child=None)
        else:
            child[True]=CreateTree(dataSet[dataSet[bestfeat]==bestVal].drop(bestfeat,axis=1),featDict,method,parent_res)
            child[False]=CreateTree(dataSet[dataSet[bestfeat]!=bestVal],featDict,method,parent_res)
    else:
        bestfeat=getBestFeature(dataSet,method)
        for label in featDict[bestfeat]:
            child[label]=CreateTree(dataSet[dataSet[bestfeat]==label].drop(bestfeat,axis=1),featDict,method,parent_res)
    
    return decisionnode(child=child,feature=bestfeat,result=parent_res,method=method,CART_val=bestVal)

def collect_feat(dataSet):
    featDict={}
    for feat in dataSet.columns[:-1]:
        featDict[feat]=list(Counter(dataSet[feat]))
    return featDict
def k_fold(dataSet,k,i):
    avg_size=dataSet.shape[0]/k
    begin=int(avg_size*(i))
    end=int(avg_size*(i+1))
    train_set=dataSet.drop(labels=range(begin,end),axis=0)
    return train_set, dataSet.loc[begin:end-1]
def classify(tree,target,featIndex=None):
    if type(target)==pd.core.series.Series:
        if tree.child!=None:
            return classify(tree.child[target[tree.feature]],target,featIndex)
        else:
            return tree.result
    else:   
        if tree.child!=None:
            return classify(tree.child[target[featIndex[tree.feature]]],target,featIndex)
        else:
            return tree.result
def classify_CART(tree,target,featIndex):
    if type(target)==pd.core.series.Series:
        if tree.child!=None:
            return classify_CART(tree.child[target[tree.feature]==tree.CART_val],target,featIndex)
        else:
            return tree.result
    else:   
        if tree.child!=None:
            return classify_CART(tree.child[target[featIndex[tree.feature]]==tree.CART_val],target,featIndex)
        else:
            return tree.result
def validate(tree,v_set):
    featIndex={}
    i=0
    for feat in v_set.columns:
        featIndex[feat]=i
        i+=1
    corrCnt=0
    if tree.method=='CART':
        for row in v_set.iterrows():
            if classify_CART(tree,list(row[1]),featIndex)==v_set['Label'][row[0]]:
                corrCnt+=1
    else:
        for row in v_set.iterrows():
            if classify(tree,list(row[1]),featIndex)==v_set['Label'][row[0]]:
                corrCnt+=1
    return corrCnt/v_set.shape[0]
def train(dataSet,k,method='C4.5'):
    accuracy=0.0
    for i in range(k):
        s1,s2=k_fold(dataSet,k,i)
        tree=CreateTree(s1,collect_feat(dataSet),method)
        currAc=validate(tree,s2)
        accuracy+=currAc
    return accuracy/k

def main(kmax=10,kmin=2):
    res={}
    ds=pd.read_csv('lab2_dataset/car_train.csv')
#	命令行参数
    methods = ['ID3','C4.5','CART']
#	为每个方法训练决策树
    for method in methods:
        res[method]=[]
#	为每个k训练决策树
        for i in range(int(kmin),int(kmax)):
            print("k=", i, ":running...\b")
#			记录准确率并加入字典中的列表
            acc=train(ds, int(i), method)
            res[method].append(acc)
            print(method,"'s accuracy",acc)
    data_df = pd.DataFrame(res)
#	保存在当前目录下的csv文件中
    data_df.to_csv('result.csv')
def test():
    ds=pd.read_csv('car_train.csv')
    ds2=pd.read_csv('2.csv')
    tree = CreateTree(ds, collect_feat(ds), 'ID3')
    featIndex={}
    i=0
    for feat in ds2.columns:
        featIndex[feat]=i
        i+=1
    for row in ds2.iterrows():
        print(classify(tree,list(row[1]),featIndex))


if __name__=="__main__":
    # command input: minimum and maximum k for training
    if(len(sys.argv)==2):
        kmax = sys.argv[1]
        main(kmax)
    if(len(sys.argv)>2):
        kmin = sys.argv[1]
        kmax = sys.argv[2]
        main(kmax,kmin)