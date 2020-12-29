from enum import IntEnum
import numpy as np
SIZE = 8

class player(IntEnum):
    BLACK=1
    WHITE=2

dirList=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

color={
    player.BLACK:-1,
    player.WHITE:1
}

def move(pos,dir):
    return (pos[0]+dir[0],pos[1]+dir[1])

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

    def game_over(self):
        """
        无空位可下
        :return:int
            0-false
            1-black win
            2-white win
            3-draw
        """
        lb,lw=len(self.black_chess),len(self.white_chess)
        if lb+lw==self.total:
            if lb!=lw:
                return 1 if lb>lw else 2
            else:
                return 3
        return 0

    def is_valid(self,pos):
        x,y=pos
        return x>=0 and x<self.size and y>=0 and y<self.size

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

    
    def add_chess(self,pos,curr_player):
        if self.is_valid(pos):
            x,y=pos
            if curr_player == player.BLACK:
                my = self.black_chess
                oppo = self.white_chess
            else:
                oppo = self.black_chess
                my = self.white_chess
            self.board[x][y]=color[curr_player]
            my.add(pos)
            self.reverse(curr_player,(pos,self.available[pos]))

    def add_tensor(self,pos,curr_player):
        (x,y)=(pos//self.size,pos%self.size)
        self.add_chess((x,y),curr_player)

    def state(self):
        return np.array(self.board, dtype=np.int).flatten()

    def Display(self):
        for n in range(self.size):
            print('----', end='')
        print('')
        for m in range(self.size):
            for n in range(self.size):
                if self.board[m][n] == color[player.BLACK]:
                    print("| x ", end='')
                elif self.board[m][n] == color[player.WHITE]:
                    print("| o ", end='')
                else:
                    print("|   ", end='')
            print('|', m)
            for n in range(self.size):
                print('----', end='')
            print('')
        for n in range(self.size):
            print('  {} '.format(n), end='')
        print('\n\n')