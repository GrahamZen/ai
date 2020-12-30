from reversi import *
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from collections import Counter

NUM_STATES=SIZE*SIZE
NUM_ACTIONS=SIZE*SIZE+1

TRANSITION_CAP=200
UPDATE_DELAY=100
BATCH_SIZE=32
GAMMA=0.9

ETA=0.0005
EPISODE=5000000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

reward_dict={
    0:0,
    1:100,
    2:-100,
    3:0
}

class NET(nn.Module):
    """定义网络结构

    Returns:
        x [tensor] -- (batch, N_ACTION)，每一行表示各个action的分数
    """

    def __init__(self):
        super(NET, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(NUM_STATES, 128),
            nn.LeakyReLU()
        )
        # self.linear1.weight.data.normal_(0, 0.1)

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 4, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(4, 8, 3, 1, 1),
            nn.LeakyReLU()
        )

        self.linear2 = nn.Sequential(
            nn.Linear(8 * 128, NUM_ACTIONS)
        )

    def forward(self, x):
        x = self.linear1(x)
        x = x.view(x.shape[0], 1, -1)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = x.flatten()
        x = x.view(x.shape[0], -1)
        x = self.linear2(x)
        return x


class AI(nn.Module):
    def __init__(self):
        super(AI, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.fcl1 = nn.Linear(1296, 100)
        self.fcl2 = nn.Linear(100, NUM_ACTIONS)

    def forward(self, x):
        x = x.view(-1,1,SIZE, SIZE)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.view(-1, 1296)
        x = F.relu(self.fcl1(x))
        x = self.fcl2(x)

        return x

class DQN(object):
    def __init__(self,turn,load=False,PATH1="model_offensive.pth",PATH2="model_defensive.pth",agent=AI):
        self.transitions = np.zeros((TRANSITION_CAP, 2 * NUM_STATES + 3))
        self.transitions_i = 0
        self.learn_iter = 0

        self.Q, self.Q_ = agent(), agent()
        if load:
            if turn == player.BLACK:
                if os.path.isfile(PATH1):
                    self.Q.load_state_dict(torch.load(PATH1))
                    print(str(turn),"'s model was loaded successfully.")
                else:
                    print("failed to load model.")
            elif turn == player.WHITE:
                if os.path.isfile(PATH2):
                    self.Q.load_state_dict(torch.load(PATH2))
                    print(str(turn),"'s model was loaded successfully.")
                else:
                    print("failed to load model.")
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=ETA)
        self.criteria = nn.MSELoss()
        self.Q.to(device)
        self.Q_.to(device)
        self.turn=turn

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

    def store_transition(self, state, action, reward, succState,not_end=0):
        self.transitions[self.transitions_i % TRANSITION_CAP] = np.hstack((state, action, reward, succState,not_end))
        self.transitions_i += 1

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
            not_end = trans_batch[:, -1]
            # 计算两个网络的y值
            y1_batch = self.Q(state_batch).gather(1, action_batch).double()
            oppo_Q_out = opponent(succState_batch).detach().max(1)[0].view(-1, 1)
            # 对方网络的输出取反作为自己网络的Q值
            y2_batch = reward_batch - GAMMA * oppo_Q_out.mul(torch.as_tensor(not_end).view(-1, 1).to(device))
            # 使用MSE作为loss函数
            loss = self.criteria(y1_batch, y2_batch)
            print(float(loss))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss

def train(load=False,save_path1="model_offensive.pth",save_path2="model_defensive.pth",agent=AI):
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
            # G.Display()
            reward=reward_dict[G.game_over()]
            succState=G.state()
            if reward!=0:
                offensive.store_transition(state,act,reward,succState,0)
            else:
                offensive.store_transition(state,act,reward,succState,1)
            roundo+=1
            if roundo > TRANSITION_CAP:
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
            if roundd > TRANSITION_CAP:
                print('Episode:{} | loss:{}'.format(episode, defensive.learn(offensive.Q_)))
                roundd=0
                break

        if (episode + 1) % 100 == 0:
            torch.save(offensive.Q.state_dict(), save_path1)
            torch.save(defensive.Q.state_dict(), save_path2)
if __name__ == '__main__':
    # torch.cuda.set_device(1)
    train("AI_o","AI_d")