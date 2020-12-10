from collections import Counter
from enum import IntEnum

cell_num = 22

avail_pos = set()
all_pos = set()
curr_pos = set()
human_pos = set()
ai_pos = set()

pattern_score = {
    (1, 1, 1, 1, 1): 99999999,
    (0, 1, 1, 1, 1, 0): 100000,
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


class player(IntEnum):
    """
    docstring
    """
    WINNER = 0
    AI = 1
    HUMAN = 2


def pos_add(pos, dir, k):
    return tuple([i+k*j for i, j in zip(pos, dir)])


def hasNeighbor(curr_pos, pos):
    for i in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]:
        if pos_add(pos, i, 1) in curr_pos:
            return True
    return False


def auto_reversed(pattern_score, t):
    tmp_t = tuple(reversed(t))
    if tmp_t in pattern_score:
        return tmp_t
    else:
        return t


def OverBound(cell_num, pos):
    xi, yi = pos
    return xi < 0 or xi >= cell_num or yi < 0 or yi >= cell_num


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


def get_matched(playerType):
    if playerType == player.HUMAN:
        opponent = ai_pos
        mine = human_pos
    else:
        opponent = human_pos
        mine = ai_pos
    matched = []
    visited = set()
    for pos in mine:
        matched += PatternMatch(pos, visited, mine, opponent)
    matched = dict(Counter(matched))

    return matched

def check_winner(pos, playerType):
    if playerType == player.HUMAN:
        search_area = human_pos
    elif playerType == player.AI:
        search_area = ai_pos
    for dir in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        for i in range(-4, 1):
            found_pattern = []
            found_pos = []
            for j in range(0, 5):
                tmp_pos = pos_add(pos, dir, i + j)
                if tmp_pos in search_area:
                    found_pattern.append(1)
                    found_pos.append(tmp_pos)
            p5 = tuple(found_pattern)
            if p5 == (1, 1, 1, 1, 1):
                return found_pos
    return []

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

    if human_matched.get((1, 1, 1, 1, 1), 0) > 0:
        return 9999999
    if ai_matched.get((1, 1, 1, 1, 1), 0) > 0:
        return - 9999999
    sum = 0
    if playerType==player.AI:
        for pattern in human_matched:
            if pattern_score[pattern] >= 100:
                sum += 1
        temp = human_matched.get((0, 1, 1, 1, 1, 0), 0)
        human_matched[(0, 1, 1, 1, 1, 0)] = sum / 2 + temp
    elif playerType==player.HUMAN:
        for pattern in ai_matched:
            if pattern_score[pattern] >= 100:
                sum+=1
        temp = ai_matched.get((0, 1, 1, 1, 1, 0), 0)
        ai_matched[(0, 1, 1, 1, 1, 0)] = sum / 2 + temp
    for pattern in human_matched:
        human_score += pattern_score[pattern]*human_matched[pattern]*(10**factor)

    for pattern in ai_matched:
        ai_score += pattern_score[pattern]*ai_matched[pattern]*(10**(-factor))

    return human_score - ai_score



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