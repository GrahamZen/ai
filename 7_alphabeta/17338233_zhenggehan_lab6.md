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

搜索策略在minimax函数中实现，搜索时遍历棋盘上的每个可以下棋的位置，如果发现该位置的8邻域没有棋子，就跳过。最大玩家和最小玩家轮流搜索自己要下的位置，为了加快搜索的速度，**按照棋面的分数对位置进行大到小排序**，当深度为0时返回，当发现对方已经赢得比赛时，也返回。

## 代码展示

为提高可读性，我设计了玩家类型的枚举，共三种，在下棋时，棋盘上打印棋子时都会使用，WINNER对应的是无关玩家的类型，以及赢家。

```python
class player(IntEnum):
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
        ai_pos.add(pos)
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

### 模式匹配函数

模式匹配函数从一个点的某个方向前5个点到后5个点进行匹配，找出棋子序列后与上面所说的模式比对，存在则**不重复的**记录下来，最终返回。

```python
def PatternMatch(target_pos, visited, mine, opponent):
    """
    找出一个点周围6个子可能的模式
    """
    match = []
    # 遍历每个方向
    for dir in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        # 向前推五个棋子的位置
        for i in range(-5, 1):
            found_pattern = []
            found_pos = []
            # 判断从起始位置开始的六个子的种类
            for j in range(0, 6):
                tmp_pos = pos_add(target_pos, dir, i + j)
                if tmp_pos in opponent or OverBound(cell_num, tmp_pos):
                    found_pattern.append(2)
                    found_pos.append(tmp_pos)
                elif tmp_pos in mine:
                    found_pattern.append(1)
                    found_pos.append(tmp_pos)
                else:
                    found_pattern.append(0)
                    found_pos.append(tmp_pos)
            # 匹配6个子的模式，找到模式则放入结果列表中，注意访问过的模式对应的棋子位置要存入集合中，避免多次统计，下面三个匹配过程都要检查是否访问过
            p6 = auto_reversed(pattern_score, tuple(found_pattern))
            found_set=set(found_pos)
            if p6 in pattern_score and not found_set.issubset(visited):
                visited.update(found_set)
                match.append(p6)
            # 取出其中5个子进行5个子的模式的匹配
            p5 = auto_reversed(pattern_score, tuple(found_pattern[:-1]))
            found_set=set(found_pos[:-1])
            if p5 in pattern_score and not found_set.issubset(visited):
                visited.update(found_set)
                match.append(p5)
            # 取出另一边的5个子进行匹配
            found_set=set(found_pos[1:])
            p5 = auto_reversed(pattern_score, tuple(found_pattern[1:]))
            if p5 in pattern_score and not found_set.issubset(visited):
                visited.update(found_set)
                match.append(p5)
    return match
```

### 评价函数

评价函数使用上面描述的阵型的字典，先判断是否有胜者，然后根据字典计算总分返回。注意下一步的玩家不同，评价的结果也是不同的，比如对于双方都有一个冲四的情况，如果先手是AI，那应该直接下在冲四的阵型上取胜，也就是说自己的分数应该有更高的权重，但是如果AI是后手，那么看到有两个冲四，就输了，这时人类的阵型分数的权重更高。这样评价可以使得AI在该主动出击的时候主动出击。

```python
def evaluate(human_matched, ai_matched, playerType=player.WINNER):
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
    ai_score = 0
	# 胜利则直接返回结果，不计算其他阵型
    if human_matched.get((1, 1, 1, 1, 1), 0) > 0:
        return 9999999
    if ai_matched.get((1, 1, 1, 1, 1), 0) > 0:
        return -9999999
    for pattern in human_matched:
        human_score += pattern_score[pattern]*human_matched[pattern]+factor*100

    for pattern in ai_matched:
        ai_score += pattern_score[pattern]*ai_matched[pattern]-factor*100

    return human_score - ai_score
```

### 单点评价函数

```python
def evaluate_single_point(pos, playerType):
    score=0
    if playerType == player.HUMAN:
        opponent = ai_pos
        mine = human_pos
    else:
        opponent = human_pos
        mine = ai_pos
    # 找出单个点周围可能存在的模式
    matched = PatternMatch(pos,set(), mine, opponent)
    matched = dict(Counter(matched))
    # 根据模式计算分数
    for pattern in matched:
        score += pattern_score[pattern] * matched[pattern]
    return score
```

### $Alpha-Beta$剪枝

代码根据算法直接编写，get_matched函数根据玩家类型找出匹配的阵型列表，evaluate函数进行打分。minimax算法是递归的，结束条件是游戏结束或到达最大深度。在遍历位置前，位置的列表要根据单点评价进行排序。

```python
def maxValue(lst_pos, depth, alpha, beta):
    if depth == 0 or check_winner(lst_pos,player.AI):
        return evaluate(get_matched(player.HUMAN),get_matched(player.AI),player.HUMAN), (-1, -1)
    currBestPos = (-1, -1)
    avail_pos = sorted(list(all_pos - curr_pos),key=lambda pos: evaluate_single_point(pos,player.HUMAN),reverse=True)
    for pos in avail_pos:
        # 没有邻居则不考虑
        if not hasNeighbor(curr_pos, pos):
            continue
        # 假设在pos处下棋
        human_pos.add(pos)
        curr_pos.add(pos)

        # 计算最大值
        new_alpha, _ = minValue(pos, depth - 1, alpha, beta)

        # 若新的更大，则更新记录的最大值
        if alpha < new_alpha:
            alpha = new_alpha
            currBestPos = pos

        # 移除刚刚下的棋子
        human_pos.remove(pos)
        curr_pos.remove(pos)

        # 剪枝
        if alpha >= beta:
            return alpha, currBestPos
        
    return alpha, currBestPos


def minValue(lst_pos, depth, alpha, beta):
    if depth == 0 or check_winner(lst_pos,player.HUMAN):
        return evaluate(get_matched(player.HUMAN),get_matched(player.AI),player.AI), (-1, -1)
    
    currBestPos = (-1, -1)
    avail_pos = sorted(list(all_pos - curr_pos),key=lambda pos: evaluate_single_point(pos,player.AI),reverse=True)
    for pos in avail_pos:
        # 没有邻居则不考虑
        if not hasNeighbor(curr_pos, pos):
            continue
        # 假设在pos处下棋
        ai_pos.add(pos)
        curr_pos.add(pos)

        # 计算最小值
        new_beta, _ = maxValue(pos, depth - 1, alpha, beta)

        # 若新的更小，则更新记录的最小值
        if beta > new_beta:
            beta = new_beta
            currBestPos = pos

        # 移除刚刚下的棋子
        ai_pos.remove(pos)
        curr_pos.remove(pos)

        # 剪枝
        if beta <= alpha:
            return beta, currBestPos

    return beta, currBestPos
```

## 实验结果以及分析

理论上只有AI需要知道各种棋面的分数，所以人类的分数我是单独打印的，评估函数按照客观玩家的角度输出分数。

第一轮AI先手：

![image-20201208224216378](E:\workspace\ai\7_alphabeta\17338233_zhenggehan_lab6.assets\image-20201208224216378.png)

第一轮人类后手：

![image-20201208224226469](E:\workspace\ai\7_alphabeta\17338233_zhenggehan_lab6.assets\image-20201208224226469.png)

第二轮AI先手：

![image-20201208224232235](E:\workspace\ai\7_alphabeta\17338233_zhenggehan_lab6.assets\image-20201208224232235.png)

第二轮人类后手

![image-20201208224243341](E:\workspace\ai\7_alphabeta\17338233_zhenggehan_lab6.assets\image-20201208224243341.png)

第三轮AI先手：

![image-20201208224255263](E:\workspace\ai\7_alphabeta\17338233_zhenggehan_lab6.assets\image-20201208224255263.png)

第三轮人类后手：

![image-20201208224310497](E:\workspace\ai\7_alphabeta\17338233_zhenggehan_lab6.assets\image-20201208224310497.png)

由于AI输出的是博弈树中搜索到的最小分数，而人类输出的是当前棋面的分数，因此AI的看起来会非常小。

# 问题与思考

本次实验因为算法有清晰的伪代码，所以实现并不困难，困难在于棋盘的评价函数，虽然有可以参考的阵型，但是实际上阵型是有无数种的，相同的阵型，阵型之外的棋子如果不同，在实战中也会有非常大的差别，比如两个以上的活二在接近的位置出现可以大概率在之后变为多个活三，从而直接取得胜利，但是活二单独出现时却几乎没有威胁。我使用的是只考虑6个棋子范围内的阵型，实际上有$3^6=729$种阵型，但是我只考虑其中几种，即便如此，深度为3的时候计算的也非常慢。

限于深度，博弈树进行搜索的结果并不会考虑到一些阵型组合的效果，因为那需要双方各下若干轮才能发现。而搜索速度慢实际上是因为剪枝的不够多，剪枝的效果实际上和搜索的顺序有关，如果一开始就能搜到最优的路线，那么其他的分支就会被剪枝，因此还需要一些启发式的评估方法，然而实际上评价函数本身就是一个启发式的评估方法，因此可以在搜索前先对位置进行评价并排序，从评分最高的点开始搜索，就更容易找到最优路线，提高剪枝的概率。