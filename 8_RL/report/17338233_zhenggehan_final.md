<center><font size="6">人工智能lab9实验报告</font></center>

<center>学号：17338233      专业：计科     姓名：郑戈涵</center>

# 强化学习
## 实验原理

强化学习是指使**智能体**在于**环境**交互过程中采取行动并最大化收益的学习方法。其中有两个概念：

* 智能体
  * 感知环境的状态和奖励反馈，并进行学习和决策
    * 学习：根据外界环境的奖励调整策略
    * 决策：根据环境的状态做出动作
* 环境
  * 智能体外部的所有事物，受智能体的动作影响，并反馈给智能体奖励

强化学习的方法有很多种，一般可以分为基于模型的动态规划方法和无模型的学习方法。基于模型的方法有策略搜索，值函数估计的方法；无模型的方法有蒙特卡洛方法和时间差分方法。有些方法可以结合两种方法中的几种，同时利用多种方法的优点。

时间差分方法(Time Difference, TD)，和蒙特卡洛(Monte Carlo)方法的差别在于，蒙特卡洛法需要在一个Episode结束后才能更新一个状态的估计值，方差较大；时间差分方法则每一步都可以进行估计值的更新。

本次实验使用DQN: Deep Q Network。是基于Q Learning的深度学习方法。

### Q Learning

Q-Learning是一种off-policy TD方法，off-policy是指学习获得的最优策略(target policy)与与环境交互获得训练样本的策略(behavior policy)不同的方法，on-policy是指两种策略相同的方法。

对于on-policy方法，训练中行为策略的表现与实际目标策略的表现相同。

对于off-policy方法，训练中行为策略的表现可能并不好，但实际目标策略的表现要优于行为策略。

Q Learning的伪代码为：

```
Initialize Q(s,a),for all s in S，a in A(s),arbitrarily,and Q(terminal,.)=0
Loop for each episode:
	Initialize S
	Loop for each step of episode:
		Choose A from S using policy derived from Q(e.g.,epsilon-greedy)
		Take action A, observe R,S'
		Q(S,A)<-Q(S,A)+alpha*[R+gamma*max(Q(S,a)-Q(S,A)))]
		S<-S'
    until S is terminal
```

更新Q的公式为：

![[公式]](https://www.zhihu.com/equation?tex=+Q%5Cleft%28+S_t%2CA_t+%5Cright%29+%5Cgets+Q%5Cleft%28+S_t%2CA_t+%5Cright%29+%2B%5Calpha+%5Cleft%5B+R_%7Bt%2B1%7D%2B%5Cgamma+%5Cunderset%7Ba%7D%7B%5Cmax%7DQ%5Cleft%28+S_%7Bt%2B1%7D%2Ca+%5Cright%29+-Q%5Cleft%28+S_t%2CA_t+%5Cright%29+%5Cright%5D+)

公式中R是奖励（已知），S是状态，A是采取的动作。

算法中使用的是Q表，用于记录state-action，每个动作后都要进行更新。

### DQN

由于Q Learning需要使用表来记录每个状态每种动作的Q值，对于许多问题，状态个数太多，无法存储在表中，所以可以使用神经网络近似Q函数。伪代码如下：

![img](E:\workspace\ai\8_RL\report\17338233_zhenggehan_final.assets\v2-07095987f15891afb6290a0a27877fc4_1440w.jpg)

简单来说，Deep Q-learning需要两个网络，其中一个是目标网络，它滞后更新。先收集智能体与环境交互的信息，单条信息的形式为(状态，动作，奖励，下个状态)，然后采样一批信息并将信息中的状态交给目标网络。使用公式计算奖励的估计值并计算Q网络输出与估计值的loss，用于更新网络。

### 黑白棋中的DQN

黑白棋中有两个智能体，因此需要两个DQN，每个DQN都需要拟合自己的Q函数，虽然是零和博弈问题，但是只要学习的次数足够多，Q函数总是可以找到某个状态下最好的动作。由于黑白棋中每个智能体都无法从自己获得下个状态的Q值，因此下个状态的Q值需要由对手提供。由于两个玩家的目标是不同的，可以将两个玩家的奖励值设为相反数，在计算Loss时可以将对方的值取反，作为下个状态的Q值，类似博弈树中的minmax方法。

## 实现过程

### 黑白棋

黑白棋需要实现的方法有：

* 获得可下棋的位置
* 翻转棋子
* 判断游戏是否结束
* 下棋，分为按照位置下和按照张量下

黑白棋的定义如下：

```python
class game(object):
    def __init__(self):
        self.size = SIZE
        self.total=self.size*self.size
        self.black_chess = set()
        self.white_chess = set()
        self.board = [[0 for j in range(self.size)] for i in range(self.size)]
        self.available = {}
        self.black_chess.add((self.size // 2 - 1, self.size // 2))
        self.black_chess.add((self.size // 2, self.size // 2 - 1))
        self.white_chess.add((self.size // 2 - 1, self.size // 2 - 1))
        self.white_chess.add((self.size // 2, self.size // 2))

        for item in self.black_chess:
            self.board[item[0]][item[1]] = color[player.BLACK]
        for item in self.white_chess:
            self.board[item[0]][item[1]] = color[player.WHITE]
```

保存了棋盘和棋子集合。

#### 获得可下棋的位置

首先，每个己方的棋子在该方向上如果有相邻的连续的对方的棋子，并且对方的棋子到底后是空的，就是可以下棋的位置。总共有8个方向。side函数在pos处给出最远的有对手连续棋子的位置。并且将遇到的对手棋子保存到集合中，如果对手的棋子的下个位置可以下棋，那么这些棋子都将被翻转。

```python
def side(self,pos,chess_set,movement):
    """
        :param pos:chess's position
        :param chess_set: chess's set
        :param movement: direction
        :return: the most far position at the direction beyond my chess with chess which will be invert
        """
    change=set()
    x,y=pos
    mx,my=movement
    moved=False
    while (x+mx,y+my) in chess_set:
        x+=mx
        y+=my
        moved=True
        change.add((x,y))
	return moved,(x,y),change
```

要找可以下棋的位置，只需要利用side函数，遍历自己下过的棋子和8个不同方向，判断最远的位置是否可以下棋，就可以找到所有可以下棋的位置。并且已知该位置可以下棋后，可以再用side函数，找到下棋后被己方棋子夹住的其他方向的棋子，将其加入集合，就可以找到所有应该翻转的棋子。

```python
def all_valid(self,curr_player):
    """
    :param curr_player:next player is me
    :return:a set of chess valid to add
    """
    if curr_player==player.BLACK:
        my=self.black_chess
        oppo=self.white_chess
    else:
        oppo=self.black_chess
        my=self.white_chess
    ret={}
    for pos in my:
        for dir in dirList:
            moved, temp, change=self.side(pos,oppo,dir)
            if moved:
                afterMove=move(temp,dir)
                if afterMove not in my and self.is_valid(afterMove):
                    for dir2 in dirList:
                        # find other chess which will be eaten by afterMove
                        moved, temp, tmp_change = self.side(afterMove, oppo, dir2)
                        if moved and move(temp,dir2) in my:
                            change.update(tmp_change)
                    ret[afterMove]=change
    self.available=ret
    return ret
```

#### 翻转

翻转函数可以利用集合很容易的实现，将翻转的棋子加入己方的棋子，从对方的棋子中去掉，自己下的棋也加入己方的棋子即可。

```python
def reverse(self,curr_player,last_pos):
    if curr_player==player.BLACK:
        my=self.black_chess
        oppo=self.white_chess
    else:
        oppo=self.black_chess
        my=self.white_chess
    pos,change=last_pos
    my |=change
    my.add(pos)
    oppo-=change
    for pos in my:
        self.board[pos[0]][pos[1]]=color[curr_player]
```

### DQN

DQN是基于一个普通的神经网络实现的

#### AI

网络的结构是参考他人效果比较好的网络，虽然自己也尝试了不少类型的网络，但是该网络效果

```python
class AI(nn.Module):
    def __init__(self):
        super(AI, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(STATE_CNT, 128),
            nn.LeakyReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 4, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(4, 8, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(8 * 128, ACTION_CNT)
        )

    def forward(self, x):
        x = self.linear1(x)
        x = x.view(x.shape[0], 1, -1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.linear2(x)
        return x

```



## 实验结果分析

![image-20201230002147896](E:\workspace\ai\8_RL\report\17338233_zhenggehan_final.assets\image-20201230002147896.png)

# 问题与思考


