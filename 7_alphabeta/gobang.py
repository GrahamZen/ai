# -*- coding=utf-8 -*-
from game_tree import *
import sys,pygame
from pygame.locals import MOUSEBUTTONUP
import pygame.gfxdraw



space = 70  # 四周留下的边距
cell_size = 40  # 每个格子大小
grid_size = cell_size * (cell_num - 1) + space * 2  # 棋盘的大小
BACKGROUND_COLOR=(212, 145, 65)
MAX_DEPTH=3

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
    pygame.display.update()

def game_loop(screen, chess_arr, turn):
    font =  pygame.font.SysFont("Comic Sans MS",20)                
    flag = 1
    pos = (-1, -1)
    ai_v = 0
    human_v=0
    while True:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if flag == turn and event.type == pygame.MOUSEBUTTONUP:  # 鼠标弹起
                x, y = pygame.mouse.get_pos()  # 获取鼠标位置
                pos = get_board_index((x, y))
                
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
                    human_v = round(evaluate(get_matched(player.HUMAN), get_matched(player.AI)), 3)
                    pygame.draw.rect(screen,BACKGROUND_COLOR,(0,0,grid_size,45),0)
                    surface = font.render('AI\'s score:' + str(ai_v), True, (255, 200, 10))
                    screen.blit(surface,(30,15))
                    surface = font.render('Human\'s score:' + str(human_v), True, (255, 200, 10))
                    screen.blit(surface,(260,15))
                    pygame.display.update()
                else:
                    continue

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
        pygame.display.update()
        
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

    add_chess(screen, [(4, 5), (5, 6), (6, 8), (8, 5), (8, 7), (3, 2), (8, 6), (8, 8), (7, 8), (9, 8), (6, 9), (5, 9), (7, 7)], player.AI)
    add_chess(screen, [(5, 5), (6, 5), (6, 7), (7, 6), (5, 4), (4, 3), (6, 3), (8, 4), (8, 9), (5, 8), (10, 8), (5, 10), (7, 4)], player.HUMAN)

    game_loop(screen, chess_arr, turn)


if __name__ == "__main__":
    game_play()