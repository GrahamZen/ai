<center><font size="6">人工智能lab1实验报告</font></center>

​																		**学号：17338233      专业：计科     姓名：郑戈涵**

# 决策树

## 算法原理

决策树是一种有监督学习的方法，其样本有自己的属性和分类的结果。学习过程中，将会根据样本的属性和分类结果不断地建立分支和节点，形成一棵树。一般通过递归的方式生成。递归的终止条件有三种情况：

* 当前节点的样本的分类结果一样
* 样本虽然分类结果不一致，但是没有属性可以用来区分样本
* 样本集合为空

测试时，决策树通过判断样本的属性不断进入分支，到叶节点时可以得到分类的结果。

由于分支取决于属性，而不同的属性顺序会导致树的长度不同，因此决定分支属性的不同方法会影响算法的效率。这次实验共有三个算法：

* ID3

  * 基于信息增益选择决策点

  * 步骤

    1. 计算数据集D的经验熵
       $$
       H(D)=-\sum_{d\in D}p(d)\log p(d)
       $$

    2. 计算特征A对数据集D的条件熵H(D|A)
       $$
       H(D|A)=\sum_{a\in A}p(a)H(D|A=a)
       $$

    3. 计算信息增益
       $$
       g(D,A)=H(D)-H(D|A)
       $$

    4. 选择信息增益最大的特征作为决策点

* C4.5

  * 基于信息增益率选择决策点

  * 步骤

    1. 计算A对数据集D的信息增益
       $$
       g(D,A)=H(D)-H(D|A)
       $$

    2. 计算数据集D关于特征A的值的熵 SplitInfo(D,A)
       $$
       SplitInfo(D,A)=-\sum_{j=1}\frac{|D_j|}{|D|}\times\log (\frac{|D_j|}{|D|})
       $$

    3. 计算信息增益率
       $$
       gRatio(D,A)=(H(D)-H(D|A))/SplitInfo(D,A)
       $$

    4. 选择信息增益率最大的特征作为决策点

* CART

  * 基于基尼系数选择决策点

  * 步骤

    1. 计算特征A的条件下，数据集D的基尼系数
       $$
       gini(D,A)=\sum_{j=1}^{v}p(A_j)\times gini(D_j|A=A_j)
       $$
       其中:
       $$
       gini(D_j|A=A_j)=\sum_{i=1}^{n}p_i(1-p_i)=1-\sum_{i=1}^{n}p_i^2
       $$
       v表示A的取值个数，n表示类别个数

    2. 选择基尼系数最小的特征作为决策点

## 伪代码或者流程图

### 主模块

```
proc decision_tree():
input:  dataSet,
        k,//交叉验证参数
        method//算法(ID3,C4.5,CART)
output tree,accuracy
	train_set, validation_set=k_fold(dataSet,k)
	tree=CreateTree(train_set)
	accuracy<-accuracy+validate(tree,validation_set)
	return accuracy/k
end proc
```

### 验证模块

```
proc validate():
input:	tree,validation_set
output: accuracy
	corrCnt=0
	for sample in validation_set:
		if classify(tree,sample)==validation_set[sample].result
			corrCnt<-corrCnt+1
		end if
    end for
	return corrCnt/validation_set.size
end proc
```

### 分类模块

```
proc classify():
input:	tree,target
output:	result
	if tree.child!=None:
		return classify(tree.child[target[tree.feature]],target)
	else:
		return tree.result
	end if
end proc
```

## 代码展示

本次编程使用python的面向对象方法，算法和决策树对象有较好的解耦。

### 条件熵

取出数据代入公式即可。

```python
from math import log
from collections import Counter
log2=lambda x:log(x)/log(2)
def rel_entropy(dataSet,feat):
# 投影取出特征和标签列
    currfeat=dataSet.loc[:,[feat,'Label']]
# 统计特征的个数
    featCount=Counter(currfeat.iloc[:,0])
    part_ent=0.0
    ent=0.0
    for feature in featCount:
        proportion=featCount[feature]/dataSet.shape[0]
        LabelCnt=Counter(currfeat.loc[currfeat[feat]==feature]['Label'])
        totalCnt=sum(LabelCnt.values())
        for r in LabelCnt.keys():
            p=float(LabelCnt[r])/totalCnt
# 条件熵公式
            part_ent=part_ent-p*log2(p)
# 加权求和
        ent+=part_ent*proportion
        part_ent=0
    return ent
```

### 经验熵

本函数同时可以用于计算C4.5的D关于某特征的熵，取出数据后代入公式即可

```python
def entropy(dataSet,feature='Label'):
    ent=0.0
    exentCnt=Counter(dataSet.loc[:,feature])
    for res in exentCnt:
        p=float(exentCnt[res])/dataSet.shape[0]
        ent-=p*log2(p)
    return ent
```

### 最佳特征（ID3,C4.5）

此函数需要区分两种方法，判断最佳特征的基准不同。

```python
def getBestFeature(dataSet,method='C4.5'):
    maxEntGain=0.0
    BestFeat=None
    entGain=0.0
    exEnt=entropy(dataSet)
    for feat in dataSet.columns[:-1]:
        entGain=exEnt-rel_entropy(dataSet,feat)
        if method=='ID3':
            pass
# 需要除以经验熵
        elif method=='C4.5':
            entGain/=entropy(dataSet,feat)
        if(maxEntGain<entGain):            
            maxEntGain=entGain
            BestFeat=feat
    return BestFeat
```

### 最佳特征（CART）

#### 特征的最佳分割值

CART使用特征的值进行分支，因此需选定一个特征的值计算基尼系数，通过比较不同值得到的基尼系数得到某个特征的最佳分割值

。

```python
def getBestSplit(dataSet,feat):
# 基尼系数小于1
    minGiniIndex=1.0
    BestSplitVal=None
# 取得当前特征对应的值的集合
    featLst=list(Counter(dataSet[feat]))
# 对每个值，计算一次基尼系数
    for featVal in featLst:
        currfeat=dataSet.loc[:,[feat,'Label']]
        featCount=Counter(currfeat.iloc[:,0])
        oneSubtractGini=0.0
        giniIndex=0.0
# 将样本按照等不等于该值分成两组
        target_group=currfeat.loc[currfeat[feat]==featVal]['Label']
        another_group=currfeat.loc[currfeat[feat]!=featVal]['Label']
        portion=target_group.size/currfeat.shape[0]
        targetCnt=Counter(target_group)
        anotherCnt=Counter(another_group)
        p1=float(targetCnt[0])/target_group.size
        p2=float(anotherCnt[0])/another_group.size
# 使用公式得到基尼系数
        giniIndex+=2*(1-p1)*p1*portion+(2*(1-p2)*p2)*(1-portion)
        if(giniIndex<minGiniIndex):
            BestSplitVal=featVal
            minGiniIndex=giniIndex
    return BestSplitVal,minGiniIndex
```

#### 获得特征函数

与ID3，C4.5不同，CART需要为每个特征的每个值都计算一次基尼系数，因此计算特征的最佳分割值时得到的基尼系数都需要用于比较，得到全局的最佳特征及特征的值。

```python
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

```

上面是所有的涉及到公式的函数。

### 决策树数据结构

由于决策树有多种算法，为了能在使用同个接口调用不同算法，数据结构需要保存算法类型（method），成员中必要的是child，feature，result，由于CART算法使用了二叉树结构，其分支推导需要一个值作为辅助，因此有CART_val。child将被赋值为字典类型，由于本次实验没有连续型特征，分类时决策树可以通过样本的标签直接索引到分支上。

```python
class decisionnode:
    def __init__(self,child,feature=None,result=None,method=None,CART_val=None):
        self.feature=feature	#当前节点对应的特征
        self.result=result		#当前的结果，当child为空时有效
        self.child=child		#存放分支
        self.CART_val=CART_val	#CART时用于进入分支
        self.method=method		#确定特征时使用的方法
```

### 构建决策树

决策树以递归形式构建，终止条件共有三种（算法原理中已描述），这三种条件可以直接返回决策树节点，其他情况需要首先按照算法分成两类，ID3，C4.5找到最佳特征后就可以通过标签直接递归到子树，CART需要找出特征的分界特征结果，将其作为节点的标签值用于分类，每个特征取值都会用来产生分支，因此树的高度较大。注意生成子树时，被分割的样本中取节点的特征取值的部分的该特征可以删去，也可以保留，因为他的基尼系数不会比其他分割方式低，但是最终各个特征都只有一种取值的样本集合需要直接处理成叶节点，删去该特征需要时间开销，因此我没有删去。

该函数中寻找特征的函数分为两种，一种处理ID3，C4.5，另一种处理CART。

``` python
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
    if dataSet.shape[1]==1:
        return decisionnode(result=parent_res,feature=None,child=None)
    bestVal=None
    if method=='CART':
        bestfeat,bestVal=getBestFeat_gini(dataSet)
		bestfeat,bestVal=getBestFeat_gini(dataSet)
        if(bestfeat==None):
            return decisionnode(result=0,feature=None,child=None)
        else:            child[True]=CreateTree(dataSet[dataSet[bestfeat]==bestVal].drop(bestfeat,axis=1),featDict,method,parent_res)
            child[False]=CreateTree(dataSet[dataSet[bestfeat]!=bestVal],featDict,method,parent_res)
    else:
        bestfeat=getBestFeature(dataSet,method)
        for label in featDict[bestfeat]:
            child[label]=CreateTree(dataSet[dataSet[bestfeat]==label].drop(bestfeat,axis=1),featDict,method,parent_res)
# 返回时设置使用的方法    
    return 
decisionnode(child=child,feature=bestfeat,result=parent_res,method=method,CART_val=bestVal)
```

### 收集特征取值

最初的做法中，我只在构建决策树时收集特征取值，但是训练时发现有些特征的取值会不在集合，导致分类时样本找不到对应的分支出错。因此特征取值需要在整个数据集生成时统计。返回的结果是主键为特征，值为特征取值的字典。

``` python
def collect_feat(dataSet):
    featDict={}
    for feat in dataSet.columns[:-1]:
        featDict[feat]=list(Counter(dataSet[feat]))
    return featDict
```

### 分类

分类是递归的过程，对于ID3，C4.5，使用测试目标的当前节点特征对应的值作为主键在字典中索引即可找到下一个决策树节点，当分支不存在时，即可返回结果。

```python
def classify(tree,target,featIndex=None):
    if tree.child!=None:
        return classify(tree.child[target[featIndex[tree.feature]]],target,featIndex)
    else:
        return tree.result
```

对于CART，只需要判断测试目标的当前节点特征对应的值和决策树节点的值是否相等，即可找到下一个分支，同样的，分支不存在时，返回结果。

```python
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
```

### 验证

验证函数需要取出每一行作为样本，使用分类函数得到结果，并与样本的标签做比较，统计正确率并返回。此处需要根据使用的方法使用不同的分类函数，方法名通过树的成员获得。

```python
def validate(tree,v_set):
# 记录列索引对应的位置
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
# ID3/C4.5
    else:
        for row in v_set.iterrows():
            if classify(tree,list(row[1]),featIndex)==v_set['Label'][row[0]]:
                corrCnt+=1
    return corrCnt/v_set.shape[0]
```

### 训练模块

这次实验开始需要自己分割数据集，我使用了k-fold方法，即将数据集分为k份，用其中k-1份来训练得到模型，剩下一份用于验证，模型评估通过计算k次准确率，取均值得到。也就是说训练时会生成k个决策树。

#### k-fold

将数据集分为k份，只需计算区间起点和终点，取出区间中的数据作为验证集，剩余的作为训练集，将两个集合即可。

```python
def k_fold(dataSet,k,i):
    avg_size=dataSet.shape[0]/k
    begin=int(avg_size*(i))
    end=int(avg_size*(i+1))
    train_set=dataSet.drop(labels=range(begin,end),axis=0)
    return train_set, dataSet.loc[begin:end-1]
```

#### 训练函数

将输入的数据集分解成训练集和测试集，构建决策树，计算准确率并输出即可。

```python
def train(dataSet,k,method='C4.5'):
    accuracy=0.0
    for i in range(k):
        s1,s2=k_fold(dataSet,k,i)
        tree=CreateTree(s1,collect_feat(dataSet),method)
        accuracy+=validate(tree,s2)
        print(i," finished")
    return accuracy/k
```

### 主模块

主模块将获取命令行参数作为k的最大值，对每个方法遍历k从2到最大值进行训练。训练后进行准确率的测试，并将得到的准确率记录到字典中最终保存在本地的csv文件里。

```python
import pandas as pd

def main():
    res={}
    ds=pd.read_csv('lab2_dataset/car_train.csv')
#	命令行参数
    k = sys.argv[1]
    methods = ['ID3', 'C4.5', 'CART']
#	为每个方法训练决策树
    for method in methods:
        res[method]=[]
#	为每个k训练决策树
        for i in range(2,int(k)):
            print("k=", i, ":running...\b")
#			记录准确率并加入字典中的列表
            acc=train(ds, int(i), method)
            res[method].append(acc)
            print(method,"'s accuracy",acc)
    data_df = pd.DataFrame(res)
#	保存在当前目录下的csv文件中
    data_df.to_csv('result.csv')
```

## 实验结果以及分析

本次实验不考虑数据集划分，则没有参数，给定的训练集和算法可以得到唯一的决策树。由于我使用了k-fold方法，调整k的大小可以得到不同的准确率结果。

下图是运行的部分结果。

![result](E:\workspace\ai\2_决策树\report\1.png)

保存在本地的csv文件中的部分内容如下（取k<20）：

| k | ID3  | C4.5 | CART |
| ---- | ---- | ---- | ---- |
|2| 0.936921296|0.94212963|0.96875|
|3| 0.953703704|0.955439815|0.9820601851851851|
|4| 0.961805556|0.963541667|0.9837962962962963|
|5| 0.966438804|0.965279383|0.9855374047080506|
|6| 0.967013889|0.966435185|0.984375|
|7| 0.968168545|0.964119962|0.9884279178245802|
|8| 0.967013889|0.965856481|0.9855324074074076|
|9| 0.962962963|0.961805556|0.9855324074074076|
|10|0.968157682|0.967001613|0.9878444683425192|
|11|0.971641758|0.969911971|0.9878401852924146|
|12|0.971064815|0.969907407|0.9890046296296297|
|13|0.970476892|0.968163416|0.9878498694288166|
|14|0.969900903|0.968748829|0.9924646884717695|
|15|0.971629185|0.97047976|0.9878410794602698|
|16|0.970486111|0.96875|0.9907407407407407|
|17|0.972215561|0.971062158|0.9895908275948704|
|18|0.96875|0.967592593|0.9890046296296295|
|19|0.971042992|0.969886254|0.9878478246899299|
|max k|17|17|14|
|max accuracy|0.972215561|0.971062158|0.992464688471770|

ID3和C4.5都始终保持93%以上的准确率，并且大部分情况在96%以上，而CART则更高，几乎达到98%。可见在选择特征时，基尼系数是一个更好的标准。

![result](E:\workspace\ai\2_决策树\report\reSult.svg)

使用表格中的信息作图后可以发现，以k=9为分界线，三种方法都有明显的从下降趋势变为上升趋势，超过16后有一定下降，我认为这是因为k=10之前的决策树是欠拟合的。k=10后训练集的样本数量较多，因此决策树接近收敛了。而k=16之后，准确率在有明显下降的同时，开始出现较大波动，这可能是因为训练集较大而过拟合，波动则可能是因为验证集的样本个数下降，导致结果的随机性增加。**k较大时的准确率没有什么意义**。不过可以注意到，虽然三个方法的准确率并不相同，变化的趋势却基本相同。ID3和C4.5甚至在训练集较大时结果基本一样，原因是该训练集的特征取值并不多，两种算法在训练样本充分时没有很大的差别。

此外，由于本次实验使用了k-fold方法，训练的时间并不短。运行时可以发现CART的运行速度较快。

## 思考题

### 决策树有哪些避免过拟合的方法？

1. 进行预剪枝或后剪枝
   * 预剪枝
     * 在决策树生成过程中进行，若产生分支不会提高验证集上的准确率，则不产生分支
   * 后剪枝
     * 在决策树生成过程中进行，若把非叶节点变为叶节点不降低在验证集上的准确率，则变成叶节点。
2. 对原训练集做预处理，筛选出能够更好反映样本特征的集合。
3. 使用随机森林算法，用随机抽样的数据生成多棵决策树，使用每棵树对样本进行分类，按照多数投票决定最终分类结果。

### C4.5相比于ID3的优点是什么，C4.5又可能有什么缺点？

#### 优点

* C4.5解决了ID3一般会优先选择有较多属性值的特征的问题。
* 可以处理连续特征

#### 缺点

* C4.5使用的仍然是多叉树，树的规模较大。
* C4.5只能用于分类
* 信息增益率的计算公式需要更多的对数运算，训练速度更慢。

### 如何用决策树来进行特征选择（判断特征的重要性）？

* ID3,C4.5，使用信息增益，信息增益（率）高的特征更重要。

* CART，基尼系数最小的特征分割点对应的特征最重要。

* 选择特征构建成树后，比较修改特征后准确率的变化，准确率提高则修改后的特征更重要。要尽量避免选择分支过多的特征，比如ID。