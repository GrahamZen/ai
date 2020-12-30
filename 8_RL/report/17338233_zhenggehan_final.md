<center><font size="6">人工智能期末大作业实验报告</font></center>

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

![image-20201231004838893](E:\workspace\ai\8_RL\report\17338233_zhenggehan_final.assets\image-20201231004838893.png)

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

DQN有两个结构相同的神经网络，一个为目标网络，一个为最新的网络。

#### AI

网络的结构是参考他人效果比较好的网络的大体结构。输入的参数为当前的棋面，64*1的向量，输出是65维的向量，包括64个位置下棋的动作和无位置可下的1个动作。

```python
class AI(nn.Module):
    def __init__(self):
        super(AI, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.fcl1 = nn.Linear(1296, 100)
        self.fcl2 = nn.Linear(100, NUM_ACTIONS)

    def forward(self, x):
        x = x.view(1,1,SIZE, SIZE)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.view(-1, 1296)
        x = F.relu(self.fcl1(x))
        x = self.fcl2(x)
        return x
```

每个DQN只能学习一种下法，默认黑先白后，因此定义一个玩家枚举。

```python
class player(IntEnum):
    BLACK=1
    WHITE=2
```

DQN在实现神经网络的基础上，对两个网络进行参数的更新。记录自己下棋的结果transitions，里面有两个state，当前的和下一步的state，各有NUM_STATES(64)个，一个action，一个reward。

```python
class DQN(object):
    def __init__(self,turn,load=False,PATH1="model_offensive.pth",PATH2="model_defensive.pth",agent=AI):
        self.transitions = np.zeros((TRANSITION_CAP, 2 * NUM_STATES + 2))
        self.transitions_i = 0
        self.learn_iter = 0

        self.Q, self.Q_ = agent(), agent()
```

#### 选择动作

该函数的输入是当前状态，输出是即将执行的动作，首先将可下棋的位置转换为索引，然后以一定概率随机选择可下棋的位置来下棋，或者将当前状态传入神经网络，神经网络得到输出后，从输出中取出可下棋的位置的概率，将最大值对应的动作返回。

```python
def select_action(self, x, G, turn, eps=0.1):
        G.all_valid(turn)
        available_pos = G.available
        if len(available_pos) == 0:
            return 64
        # 将可下棋的位置转为索引
        available_pos = list(map(lambda pos: G.size * pos[0] + pos[1], available_pos))
        if np.random.uniform() < eps:
			# 随机选一个
            action = np.random.choice(available_pos, 1)[0]
        else:
            x = torch.as_tensor(x, dtype=torch.float).view(1, -1).to(device)
            # 传入网络
            action_v = self.Q(x)[0]
            ava_action = torch.as_tensor(action_v[available_pos])
			# 取出最大的位置
            _, action_i = torch.max(ava_action, 0)

            action = available_pos[action_i]
        return action
```

#### Replay Buffer

对回放容量取余，就可以不断的更新有限容量的回放，用于网络学习。由于更新网络的公式根据下一局是否终止有不同，因此要进行记录。

```python
def store_transition(self, state, action, reward, succState,not_end=0):
    self.transitions[self.transitions_i % TRANSITION_CAP] = np.hstack((state, action, reward, succState,not_end))
    self.transitions_i += 1
```

#### 学习

学习时是向对手的网络学习，因为自己无法取得下个状态的Q值。从回放中随机挑选一批，取出state,action,reward,下个state。将state传给网络，得到每个action对应的Q值，取出自己采取的action对应的Q值，再将下个state交给对手的网络，将对方的Q值取反当做自己的下个state的Q值，这样就可以使用DQN的公式进行更新了。

每次学习时还要将目标网络进行更新。更新的函数有两种情况，对对方的网络的输出使用not_end作为掩码，才能正确的更新Q值。要注意的是tensor的类型要对应，否则会出现CUDA的异常错误。

```python
def learn(self, opponent):
    self.Q_.load_state_dict(self.Q.state_dict())
    for step in range(UPDATE_DELAY):
        self.learn_iter += 1
		# 随机选一批回放记录
        sample_index = np.random.choice(TRANSITION_CAP, BATCH_SIZE)
        trans_batch = self.transitions[sample_index, :]

        state_batch = torch.as_tensor(trans_batch[:, :NUM_STATES], dtype=torch.float).to(device)
        action_batch = torch.as_tensor(trans_batch[:, NUM_STATES:NUM_STATES + 1], dtype=int).to(device)
        reward_batch = torch.as_tensor(trans_batch[:, NUM_STATES + 1:NUM_STATES + 2], dtype=torch.float).to(device)
        succState_batch = torch.as_tensor(trans_batch[:, NUM_STATES + 2:-1], dtype=torch.float).to(device)
        not_end=trans_batch[:,-1]
        # 计算两个网络的y值
        y1_batch = self.Q(state_batch).gather(1,action_batch).double()
        oppo_Q_out = opponent(succState_batch).detach().max(1)[0].view(-1, 1)
        # 对方网络的输出取反作为自己网络的Q值
        y2_batch = reward_batch - GAMMA * oppo_Q_out.mul(torch.as_tensor(not_end).view(-1,1).to(device))
        # 使用MSE作为loss函数
        loss = self.criteria(y1_batch, y2_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    return loss
```

#### 训练

训练是两个agent互相下棋的过程，每下一步棋都将当前状态的特征向量保存起来，如果积累了一定量的记录则可以进行学习。学习时使用的是对手的Q_网络。每下一步都要进行记录，同时要记录是否有胜利者，用于网络更新。

```python
def train():
    offensive=DQN(player.BLACK)
    defensive=DQN(player.WHITE)

    for episode in range(EPISODE):
        G=game()
        roundo=0
        roundd=0
        while True:
            state=G.state()
            act=offensive.select_action(state,G,player.BLACK)
            G.add_tensor(act,player.BLACK)
            reward=reward_dict[G.game_over()]
            succState=G.state()
            if reward!=0:
                offensive.store_transition(state,act,reward,succState,0)
            else:
                offensive.store_transition(state,act,reward,succState,1)
            roundo+=1
            if roundo > 100:
                print('Episode:{} | loss:{}'.format(episode, offensive.learn(defensive.Q_)))
                roundo=0
                break

            state=G.state()
            act=defensive.select_action(state,G,player.WHITE)
            G.add_tensor(act,player.WHITE)
            reward = reward_dict[G.game_over()]
            succState = G.state()

            if reward!=0:
                defensive.store_transition(state,act,reward,succState,0)
            else:
                defensive.store_transition(state,act,reward,succState,1)
            roundd+=1
            if roundd > 100:
                print('Episode:{} | loss:{}'.format(episode, defensive.learn(offensive.Q_)))
                roundd=0
                break
```

## 实验结果分析

下面是经过10000个episode后得到的结果，与[4399](http://www.4399.com/flash/159743_1.htm)上的3星难度bot进行对局。AI是黑棋。

![image-20201230002147896](E:\workspace\ai\8_RL\report\17338233_zhenggehan_final.assets\image-20201230002147896.png)

可以明显看到，AI会主动下边缘的位置。

下图是白棋后手，可以看到，左边的3明显是最优的位置

![image-20201230184112571](E:\workspace\ai\8_RL\report\17338233_zhenggehan_final.assets\image-20201230184112571.png)

很快AI就选择了下左边的3。

![image-20201230184149884](E:\workspace\ai\8_RL\report\17338233_zhenggehan_final.assets\image-20201230184149884.png)

同样，在下图中，上面的两个3都是很理想的位置。

![image-20201230184257600](E:\workspace\ai\8_RL\report\17338233_zhenggehan_final.assets\image-20201230184257600.png)

很快AI就选择了在左边下棋，

![image-20201230184347262](E:\workspace\ai\8_RL\report\17338233_zhenggehan_final.assets\image-20201230184347262.png)

对于下面的局面，右边的2对白棋稍有优势，因此白棋也选择下这里。

![image-20201231001742991](E:\workspace\ai\8_RL\report\17338233_zhenggehan_final.assets\image-20201231001742991.png)

![image-20201231001825824](E:\workspace\ai\8_RL\report\17338233_zhenggehan_final.assets\image-20201231001825824.png)

下图的局面下，同样是吃三个子，右上角明显更重要，因此AI选择下角落。

![image-20201231002129104](E:\workspace\ai\8_RL\report\17338233_zhenggehan_final.assets\image-20201231002129104.png)

![image-20201231002219460](E:\workspace\ai\8_RL\report\17338233_zhenggehan_final.assets\image-20201231002219460.png)

虽然对部分棋面AI并不能做最优的动作，但是经过很长时间的训练，AI已经对较为重要的动作有一定的了解。

# 问题与思考

本次实验结合minmax的方法，为DQN提供了计算Q网络对下一状态Q值的模拟。

实际上DQN作为TD的方法，对于棋类游戏并非最优，因为对于难以获得很好的评价函数的棋类游戏，往往要到游戏结束才能得到奖励，对于学习是不方便的，这也导致DQN的训练非常慢，短时间内不会得到很好的模型，并且loss一直在波动，因为一旦学到一个新局面的有利动作，Q网络的MSE就会增大，然后再逐渐减小。总体上能力是螺旋上升的。

![train](E:\workspace\ai\8_RL\report\17338233_zhenggehan_final.assets\train.svg)

MCTS是一个比较适合棋类的方法，可以通过少量的模拟得到较好的结果，利用深度神经网络结合策略梯度来选择走MCTS的分支，可以得到非常好的决策结果，但是模型也会更复杂，方差也比较大。如果对黑白棋的规则足够熟悉，利用评价函数来指导学习是更快捷的方法。