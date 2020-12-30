# -*- coding=utf-8 -*-
from reversi import *
from train import *
import sys,pygame
from pygame.locals import MOUSEBUTTONUP
import pygame.gfxdraw

chessColor = {
    "white":(225, 225, 225),
    "black": (30, 30, 30),
}

space = 70  # 四周留下的边距
cell_size = 60  # 每个格子大小
grid_size = cell_size * (SIZE - 1) + space * 2  # 棋盘的大小
BACKGROUND_COLOR=(212, 145, 65)
MAX_DEPTH=2

def get_board_index(pos):
    xi = int(round((pos[0] - space)*1.0/cell_size))  # 获取到x方向上取整的序号
    yi = int(round((pos[1] - space)*1.0/cell_size))  # 获取到y方向上取整的序号
    return (xi,yi)


def draw_chess(screen, pos, chessType,avail=False):
    xi, yi = pos
    pygame.gfxdraw.aacircle(screen, xi*cell_size+space, yi*cell_size+space, 16,chessColor[chessType])    
    if not avail:
        pygame.gfxdraw.filled_circle(screen, xi * cell_size + space, yi * cell_size + space, 16, chessColor[chessType])


def show_winner(screen,winner):
    winner=str(winner).split('.')[1] +' win!'
    font =  pygame.font.SysFont("Comic Sans MS",40)                
    surface = font.render(winner, True, (152,251,152))
    screen.blit(surface, (grid_size / 2-len(winner)*11, grid_size - 60))
    pygame.display.update()

def game_update(screen, G,turn):
    screen.fill(BACKGROUND_COLOR)
    for x in range(0, cell_size*SIZE, cell_size):
        pygame.draw.aaline(screen, (220, 220, 220), (x+space, 0+space),
                        (x+space, cell_size*(SIZE-1)+space), 1)
    for y in range(0, cell_size*SIZE, cell_size):
        pygame.draw.aaline(screen, (220, 220, 220), (0+space, y+space),
                        (cell_size*(SIZE - 1) + space, y + space), 1)
    for pos in G.black_chess:
        draw_chess(screen,pos,"black")
    for pos in G.white_chess:
        draw_chess(screen,pos,"white")
    font = pygame.font.SysFont("Comic Sans MS", 20)
    for pos in G.available:
        color ="black" if turn==player.BLACK else "white"
        surface = font.render(str(len(G.available[pos])), True, color)
        screen.blit(surface,(pos[0] * cell_size + space-5, pos[1] * cell_size + space-15))
        draw_chess(screen,pos,color,True)


def game_loop(screen, G,ai,turn):
    G.all_valid(turn)
    flag=turn
    game_update(screen, G,flag)
    while True:        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if flag == turn:  # 鼠标弹起
                pos=ai.select_action(G.state(),G,turn)
                G.add_tensor(pos,flag)
                flag = 3-flag
                G.all_valid(flag)
                if len(G.available) == 0:
                    res = G.game_over()
                    if res != 0:
                        return res
                    print("no place to move")
                    flag = 3 - flag
                    G.all_valid(flag)
                game_update(screen,G,flag)
                # pygame.draw.rect(screen,BACKGROUND_COLOR,(0,0,grid_size,45),0)
                # surface = font.render('WHITE\'s score:' + str(ai_v), True, (255, 200, 10))
                # screen.blit(surface,(30,15))
                # surface = font.render('BLACK\'s score:' + str(human_v), True, (255, 200, 10))
                # screen.blit(surface,(260,15))
                pygame.display.update()

            if flag == 3-turn:  # 鼠标弹起
                if event.type == pygame.MOUSEBUTTONUP:
                    x, y = pygame.mouse.get_pos()  # 获取鼠标位置
                    pos = get_board_index((x, y))
                else:
                    continue
                if pos in G.available.keys():
                    G.add_chess(pos,flag)
                    flag = 3-flag
                    G.all_valid(flag)
                    if len(G.available) == 0:
                        res = G.game_over()
                        if res != 0:
                            return res
                        if res==3:
                            print("draw game")

                        print("no place to move")
                        flag = 3 - flag
                        G.all_valid(flag)
                    game_update(screen,G,flag)
                    # pygame.draw.rect(screen,BACKGROUND_COLOR,(0,0,grid_size,45),0)
                    # surface = font.render('WHITE\'s score:' + str(ai_v), True, (255, 200, 10))
                    # screen.blit(surface,(30,15))
                    # surface = font.render('BLACK\'s score:' + str(human_v), True, (255, 200, 10))
                    # screen.blit(surface,(260,15))
                    pygame.display.update()
                else:
                    continue

        pygame.display.update()

def show_winner(screen,winner):
    if winner==3:
        winner="draw game"
    else:
        winner=str(player(winner)).split('.')[1] +' win!'
    font =  pygame.font.SysFont("Comic Sans MS",40)
    surface = font.render(winner, True, (152,251,152))
    screen.blit(surface, (grid_size / 2-len(winner)*11, grid_size - 60))
    pygame.display.update()

def game_play():
    g = game()
    if len(sys.argv)==2:
        turn =int(sys.argv[1])
    else:
        turn =int(input('黑先手:1,白先手:2,请输入:'))
    turn = player(turn)
    pygame.init()

    screencaption = pygame.display.set_caption('黑白棋')
    screen = pygame.display.set_mode((grid_size, grid_size))  # 设置窗口长宽
    pygame.display.update()
    path = input("model path:")
    if len(path)!=0:
        ai = DQN(turn,True,path,path,agent=NET)
    else:
        ai = DQN(turn, True, agent=NET)
    res = game_loop(screen, g,ai,turn)
    show_winner(screen,res)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

if __name__ == "__main__":
    game_play()