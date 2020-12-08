<center><font size="6">人工智能lab9实验报告</font></center>

<center>学号：17338233      专业：计科     姓名：郑戈涵</center>

# 博弈树搜索

## 算法原理

### 零和博弈问题

#### 前提

* 两个博弈玩家
* 游戏和决策值都可以映射到离散空间
* 只有有限个状态和可能的决策结果
* 确定性
* 零和性：一方的损失等于另一方的收益
* 信息完备性：博弈双方知道所处状态的所有信息

#### 概念

状态集合：$S$

初始状态：$I\in S$

终止位置：$T\subset S$

后继：下一可能状态的集合

效益值（utility）：$V:\mapsto \R$，某个状态在博弈中对某个玩家的有益程度

### $MiniMax$策略

认为双方玩家都会使得状态朝对自己最有利的方向发展。

### $Alpha-Beta$剪枝

由于遍历树的复杂度与深度成指数关系，一般会进行$Alpha-Beta$剪枝。对已知无法找到更大（或更小）的效益值的分支进行剪枝操作。

## 伪代码

### $MiniMax$策略

```
DFMiniMax(n,Player)
If n is TERMINAL
	RETURN V(n)
Endif

ChildList = n.Successors(Player)
If Player == MIN
	return minimum of DFMiniMax(c,MAX) over c in ChildList
Else
	return maximum of DFMiniMax(c,MIN) over c in ChildList
Endif
```

### $Alpha-Beta$剪枝

```
AlphaBeta (n,Player,alpha,beta)
If n is TERMINAL
	retrun V(n)
n.Successprd(Player)
Endif

If Player == MAX
	for c in ChildList
		alpha = max(alpha,AlphaBeta(c,Min,alpha,beta))
		If beta<=alpha
			return alpha
		Endif
Else If Player == MIN
	for c in ChildList
		If beta<=alpha
			return beta
		Endif
Endif
```
## 实现细节

### 评价函数

一个棋盘状态的评价函数与阵型相关，参考网上的资料，我将以下阵型（除了下图还有几个）用于计算效益值。

<center class="half">
    <img src="17338233_zhenggehan_lab6.assets\2019052219344927.png" width="180"/>
    <img src="17338233_zhenggehan_lab6.assets\20190522193648513.png" width="180"/>
    <img src="17338233_zhenggehan_lab6.assets\20190522194317341.png" width="180"/>
</center>





<center class="half">
    <img src="17338233_zhenggehan_lab6.assets\2019052219433423.png" width="180"/>
    <img src="17338233_zhenggehan_lab6.assets\20190522194433545.png" width="180"/>
    <img src="17338233_zhenggehan_lab6.assets\20190522195109633.png" width="180"/>
</center>



<center class="half">
    <img src="17338233_zhenggehan_lab6.assets\20190522195214449.png" width="180"/>
    <img src="17338233_zhenggehan_lab6.assets\20190522195554569.png" width="180"/>
    <img src="17338233_zhenggehan_lab6.assets\2019052219563273.png" width="180"/>
</center>

为每个阵型定义一个分值

```python
pattern_score = {
    (1, 1, 1, 1, 1): 99999999,
    (0, 1, 1, 1, 1, 0): 10000,
    (0, 1, 1, 0, 1, 0): 5000,
    (0, 1, 1, 1, 0): 1000,
    (0, 1, 1, 1, 1): 100,
    (1, 1, 1, 0, 1): 100,
    (1, 1, 0, 1, 1): 100,
    (0, 1, 1, 0, 1, 0): 100,
    (0, 0, 1, 1, 1): 10,
    (1, 1, 0, 1, 0): 10,
    (1, 1, 0, 0, 1): 10,
    (1, 0, 1, 0, 1): 10,
    (0, 1, 1, 0, 0): 1,
}
```

为AI和人类玩家分别计算各种阵型对应的分值之和，并且根据当前的玩家类型增加或扣除一定分数，然后返回。

### 搜索策略

搜索策略在minimax函数中实现，搜索时遍历棋盘上的每个可以下棋的位置，如果发现该位置的8邻域没有棋子，就跳过。最大玩家和最小玩家轮流搜索自己要下的位置，当深度为0时返回，当发现对方已经赢得比赛时，也返回。

## 代码展示

为提高可读性，我设计了玩家类型的枚举，共三种，在下棋时，棋盘上打印棋子时都会使用，WINNER对应的是无关玩家的类型，以及赢家。

```python
class player(IntEnum):
    """
    docstring
    """
    WINNER = 0
    AI = 1
    HUMAN = 2
```

### UI设计

我使用pygame库进行界面的显示，逻辑是在while循环中更新对象属性以及响应事件。每次下棋后，程序都会在界面上绘制棋子，draw_chess函数如下：

```python
def draw_chess(screen, pos, playerType):
    if playerType == player.AI:
        chess_color = (30, 30, 30)
    elif playerType == player.HUMAN:
        chess_color = (225, 225, 225)
    else:
        chess_color=(255,0,0)
    xi, yi = pos
    if not OverBound(cell_num,pos) and (playerType==player.WINNER or (xi, yi) not in curr_pos):
        pygame.gfxdraw.filled_circle(screen, xi*cell_size+space, yi*cell_size+space, 16,chess_color)
        pygame.gfxdraw.aacircle(screen, xi*cell_size+space, yi*cell_size+space, 16,(225,225,225))    
        return True
    else:
        return False
```

函数会检查下棋的位置是否合法，将检查结果返回

### 渲染循环

人和AI的代码大部分是一样的，以AI的为例，

```python
if flag == 1 - turn:
    # ai为最小玩家
    ai_v, pos = minValue(pos, MAX_DEPTH, float('-inf'), float('inf'))
    ai_v = round(ai_v, 3)
    # 检查下的位置是否有效
    if (draw_chess(screen, pos, player.AI)):
        # 将棋子位置放入集合中
        machine_pos.add(pos)
        # 检查是否有玩家胜出，找出胜出的五个子
        win_pos = check_winner(pos, player.AI)

        if win_pos != []:
            for p in win_pos:
                # 将胜者的五个子画出
                draw_chess(screen, p, player.WINNER)
                # 显示胜者
                show_winner(screen, player.AI)
                turn=-1
                curr_pos.add(pos)
                print('ai:',pos)
                flag = 1 - flag
                pygame.draw.rect(screen,BACKGROUND_COLOR,(0,0,grid_size,45),0)                
                surface = font.render('AI\'s score:' + str(ai_v), True, (255, 200, 10))
                screen.blit(surface,(30,15))
                surface = font.render('Human\'s score:' + str(human_v), True, (255, 200, 10))
                screen.blit(surface,(260,15))
                pygame.display.update()
                else:
                    # 无效，说明无处可下，胜者为人
                    show_winner(screen, player.HUMAN)
                    turn=-1
```

### 估值函数

```python
def evaluate(human_matched, machine_matched, playerType=player.WINNER):
    """
    下一步为playerType下
    """
    if playerType == player.HUMAN:
        factor = 1
    elif playerType == player.AI:
        factor = -1    
    else:
        factor = 0

    human_score = 0
    machine_score = 0

    if human_matched.get((1, 1, 1, 1, 1), 0) > 0:
        return 9999999
    if machine_matched.get((1, 1, 1, 1, 1), 0) > 0:
        return -9999999
    for pattern in human_matched:
        human_score += pattern_score[pattern]*human_matched[pattern]+factor*100

    for pattern in machine_matched:
        machine_score += pattern_score[pattern]*machine_matched[pattern]-factor*100

    return human_score - machine_score
```



### $Alpha-Beta$剪枝

代码根据算法直接编写，get_matched函数根据玩家类型找出匹配的阵型列表，evaluate函数进行打分。

```python
def maxValue(lst_pos, depth, maxAlpha, minBeta):
    if depth == 0 or check_winner(lst_pos,player.AI):
        return evaluate(get_matched(player.HUMAN),get_matched(player.AI)), (-1, -1)
    currMaxAlpha = float('-inf')
    currBestPos = (-1, -1)
    avail_pos = all_pos - curr_pos
    for pos in avail_pos:
        if not hasNeighbor(curr_pos, pos):
            continue
        # 假设在pos处下棋
        human_pos.add(pos)
        curr_pos.add(pos)

        # 计算最大值
        currAlpha, _ = minValue(pos, depth - 1, maxAlpha, minBeta)

        # 若新的更大，则更新记录的最大值
        if currMaxAlpha < currAlpha:
            currMaxAlpha = currAlpha
            currBestPos = pos

        # 移除刚刚下的棋子
        human_pos.remove(pos)
        curr_pos.remove(pos)

        # 剪枝
        if currMaxAlpha > minBeta:
            return currMaxAlpha, currBestPos
        
        # 更新当前节点的Alpha值，用于该节点其他分支的剪枝操作
        if currMaxAlpha > maxAlpha:
            maxAlpha = currMaxAlpha
    return currMaxAlpha, currBestPos
```



## 实验结果以及分析

下面是UCS算法对迷宫的运行结果，绿色线代表路径，黄色的为搜索过的路径，有颜色的点都是迷宫的可行点，蓝色为未被搜索的路径。可以看出，大部分路径都已经被搜索过了，效率比较低。打印的结果为：

```
空间复杂度:9, 时间复杂度:274
```

![UCS](E:\workspace\ai\6_serach\report\17338233_zhenggehan_lab6.assets\UCS.svg)

# 思考题

## 策略的优缺点

| 策略 | 优点|缺点 |
| ------------|------|------|
| 一致代价搜索  | 保证完备性，最优性的前提下容易实现 | 空间复杂度较高，盲目搜索所以效率低 |
| A*搜索        | 在一致代价搜索的基础上加入启发式函数，搜索速度较快 | 空间复杂度较高，且搜索速度取决于启发式函数的特点，而且有时候难以找到满足两个性质的启发式函数 |
| 迭代加深搜索  | 空间复杂度低 | 深度是随迭代逐渐增加的，迭代重新搜索时已访问的点仍会被重新访问。开销不一致时需要满足一定条件才有最优性 |
| IDA*          | 在迭代加深搜索的基础上加入启发式函数，搜索效率提高 | 和迭代加深搜索一样，有时候难以找到满足两个性质的启发式函数 |
| 双向搜索      | 搜索深度减半，效率提高 | 需要维护两个边界集合，空间复杂度高 |

## 适用场景

| 策略 |适用场景 |
| ------------|------------|
| 一致代价搜索  | 预估搜索深度较低，或者对空间复杂度要求较低时比较适合，最短和最小问题 |
| A*搜索        | 有较好的启发式函数，大致知道或者可以确定目标位置时 |
| 迭代加深搜索  | 空间复杂度要求较高的场景 |
| IDA*          | 空间复杂度要求较高，而且有较好的启发式函数，大致知道或者可以确定目标位置的场景 |
| 双向搜索      | 已知终点和起点的场景 |
