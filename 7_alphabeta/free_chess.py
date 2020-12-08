# -*- coding=utf-8 -*-
from game_tree import *
import sys,pygame
from pygame.locals import MOUSEBUTTONUP
import pygame.gfxdraw



space = 60  # 四周留下的边距
cell_size = 40  # 每个格子大小
grid_size = cell_size * (cell_num - 1) + space * 2  # 棋盘的大小
BACKGROUND_COLOR=(212, 145, 65)
MAX_DEPTH=3


# def get_one_dire_num(lx, ly, dx, dy, m):
#     tx = lx
#     ty = ly
#     s = 0
#     while True:
#         tx += dx
#         ty += dy
#         if tx < 0 or tx >= cell_num or ty < 0 or ty >= cell_num or m[ty][tx] == 0:
#             return s
#         s += 1


# def check_win(chess_arr, turn):
#     # 先定义一个15*15的全0的数组,不能用[[0]*cell_num]*cell_num的方式去定义因为一位数组会被重复引用
#     m = [[0]*cell_num for i in range(cell_num)]
#     for x, y, c in chess_arr:
#         if c == turn:
#             m[y][x] = 1  # 上面有棋则标1
#         lx = chess_arr[-1][0]  # 最后一个子的x
#         ly = chess_arr[-1][1]  # 最后一个子的y
#         dire_arr = [[(-1, 0), (1, 0)], [(0, -1), (0, 1)], [(-1, -1), (1, 1)],
#                     [(-1, 1), (1, -1)]]  # 4个方向数组,往左＋往右、往上＋往下、往左上＋往右下、往左下＋往右上，4组判断方向

#     for dire1, dire2 in dire_arr:
#         dx, dy = dire1
#         num1 = get_one_dire_num(lx, ly, dx, dy, m)
#         dx, dy = dire2
#         num2 = get_one_dire_num(lx, ly, dx, dy, m)
#         if num1 + num2 + 1 >= 5:
#             return True

#     return False


def get_board_index(pos):
    xi = int(round((pos[0] - space)*1.0/cell_size))  # 获取到x方向上取整的序号
    yi = int(round((pos[1] - space)*1.0/cell_size))  # 获取到y方向上取整的序号
    return (xi,yi)


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


def show_winner(screen,winner):
    winner=str(winner).split('.')[1] +' win!'
    font =  pygame.font.SysFont("Comic Sans MS",40)                
    surface = font.render(winner, True, (152,251,152))
    screen.blit(surface, (grid_size / 2-len(winner)*11, grid_size - 60))
    pygame.display.update()  # 必须调用update才能看到绘图显示

def game_loop(screen, chess_arr, turn):
    flag = 1
    pos=(-1,-1)
    while True:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if flag == turn and event.type == pygame.MOUSEBUTTONUP:  # 鼠标弹起
                x, y = pygame.mouse.get_pos()  # 获取鼠标位置
                pos=get_board_index((x,y))
                if (draw_chess(screen, pos, player.HUMAN)):                
                    human_pos.add(pos)
                    win_pos = check_winner(pos, player.HUMAN)
                    if win_pos != []:
                        for p in win_pos:
                            draw_chess(screen,p,player.WINNER)
                        show_winner(screen, player.HUMAN)
                        turn = -1
                    curr_pos.add(pos)
                    print('human:',pos)
                    flag = 1 - flag
                    pygame.display.update()  # 必须调用update才能看到绘图显示

                else:
                    continue

            if flag == 1-turn and event.type == pygame.MOUSEBUTTONUP:  # 鼠标弹起
                x, y = pygame.mouse.get_pos()  # 获取鼠标位置
                pos=get_board_index((x,y))
                if (draw_chess(screen, pos, player.AI)):                
                    human_pos.add(pos)
                    win_pos = check_winner(pos, player.AI)
                    if win_pos != []:
                        for p in win_pos:
                            draw_chess(screen,p,player.WINNER)
                        show_winner(screen, player.AI)
                        turn = -1
                    curr_pos.add(pos)
                    print('AI:',pos)
                    flag = 1 - flag
                    pygame.display.update()  # 必须调用update才能看到绘图显示

                else:
                    continue
        pygame.display.update()  # 必须调用update才能看到绘图显示
        
def add_chess(screen,pos, playerType):
    if type(pos)==type([1,1]):
        for p in pos:
            add_chess(screen, p, playerType)
        return
    field = human_pos if playerType == player.HUMAN else machine_pos
    draw_chess(screen, pos, playerType)
    field.add(pos)
    curr_pos.add(pos)

def game_play():
    chess_arr = []
    if len(sys.argv)==2:
        turn =int(sys.argv[1])
    else:
        turn =int(input('人先手:1,AI先手:0,请输入:'))
    
    for i in range(cell_num):
        for j in range(cell_num):
            all_pos.add((i,j))

    pygame.init()

    screencaption = pygame.display.set_caption('五子棋')
    screen = pygame.display.set_mode((grid_size, grid_size))  # 设置窗口长宽
                
    screen.fill(BACKGROUND_COLOR)

    for x in range(0, cell_size*cell_num, cell_size):
        pygame.draw.aaline(screen, (220, 220, 220), (x+space, 0+space),
                        (x+space, cell_size*(cell_num-1)+space), 1)
    for y in range(0, cell_size*cell_num, cell_size):
        pygame.draw.aaline(screen, (220, 220, 220), (0+space, y+space),
                        (cell_size*(cell_num-1)+space, y+space), 1)

    add_chess(screen, [(4, 5), (5, 6),(6, 7),(3, 4),(2, 5)], player.HUMAN)
    add_chess(screen, [(5, 5), (6, 5),(7, 8),(2, 3),(1, 6)], player.AI)
    
    

    # draw_chess(screen,(4,5),player.AI)
    # draw_chess(screen,(5,6),player.AI)
    # draw_chess(screen,(5,5),player.HUMAN)
    # draw_chess(screen,(6,5),player.HUMAN)
    # curr_pos.update([(4, 5), (5, 6), (5, 5), (6, 5)])
    # machine_pos.update([(4, 5), (5, 6)])
    # human_pos.update([(5,5),(6,5)])
    game_loop(screen, chess_arr, turn)


if __name__ == "__main__":
    game_play()