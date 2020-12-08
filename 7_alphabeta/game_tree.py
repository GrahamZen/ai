from collections import Counter
from enum import IntEnum

cell_num = 11

avail_pos = set()
all_pos = set()
curr_pos = set()
human_pos = set()
machine_pos = set()

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
    match = []
    for dir in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        for i in range(-5, 1):
            found_pattern = []
            found_pos = []
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
            p6 = auto_reversed(pattern_score, tuple(found_pattern))
            found_set=set(found_pos)
            if p6 in pattern_score and not found_set.issubset(visited):
                visited.update(found_set)
                match.append(p6)

            p5 = auto_reversed(pattern_score, tuple(found_pattern[:-1]))
            found_set=set(found_pos[:-1])
            if p5 in pattern_score and not found_set.issubset(visited):
                visited.update(found_set)
                match.append(p5)

            found_set=set(found_pos[1:])
            p5 = auto_reversed(pattern_score, tuple(found_pattern[1:]))
            if p5 in pattern_score and not found_set.issubset(visited):
                visited.update(found_set)
                match.append(p5)
    return match


def get_matched(playerType):
    if playerType == player.HUMAN:
        opponent = machine_pos
        mine = human_pos
    else:
        opponent = human_pos
        mine = machine_pos
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
        search_area = machine_pos
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
    
    # if human_matched.get((0, 1, 1, 1, 1, 0), 0) > 0:
    #     return 9999999
        
    # if human_matched.get((0, 1, 1, 1, 1), 0) + human_matched.get((0, 1, 1, 1, 0), 0) > 1:
    #     return 9999999
    # elif machine_matched.get((0, 1, 1, 1, 1, 0), 0) > 0:
    #     return -9999999
    # elif machine_matched.get((0, 1, 1, 1, 1), 0) + machine_matched.get((0, 1, 1, 1, 0), 0) > 1:
    #     return -9999999

    for pattern in human_matched:
        human_score += pattern_score[pattern]*human_matched[pattern]+factor*100

    for pattern in machine_matched:
        machine_score += pattern_score[pattern]*machine_matched[pattern]-factor*100

    return human_score - machine_score



def maxValue(lst_pos, depth, maxAlpha, minBeta):
    if depth == 0 or check_winner(lst_pos,player.AI):
        return evaluate(get_matched(player.HUMAN),get_matched(player.AI),player.HUMAN), (-1, -1)
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


def minValue(lst_pos, depth, maxAlpha, minBeta):
    if depth == 0 or check_winner(lst_pos,player.HUMAN):
        return evaluate(get_matched(player.HUMAN),get_matched(player.AI),player.AI), (-1, -1)
    
    currMinBeta = float('inf')
    currBestPos = (-1, -1)
    avail_pos = all_pos - curr_pos
    for pos in avail_pos:
            
        if not hasNeighbor(curr_pos, pos):
            continue
        machine_pos.add(pos)
        curr_pos.add(pos)

        currBeta, _ = maxValue(pos, depth - 1, maxAlpha, minBeta)

        if currMinBeta > currBeta:
            currMinBeta = currBeta
            currBestPos = pos

        machine_pos.remove(pos)
        curr_pos.remove(pos)

        if currMinBeta < maxAlpha:
            return currMinBeta, currBestPos

        if currMinBeta < minBeta:
            minBeta = currMinBeta
    return currMinBeta, currBestPos
