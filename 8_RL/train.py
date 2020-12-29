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
BATCH_SIZE=1
GAMMA=0.9

ETA=0.001
EPISODE=5000000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

reward_dict={
    0:0,
    1:100,
    2:-100,
    3:0
}

class AI(nn.Module):
    def __init__(self):
        super(AI, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(NUM_STATES, 128),
            nn.LeakyReLU()
        )
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
        x = x.view(x.shape[0], -1)
        x = self.linear2(x)
        return x

class Net1(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.fcl1 = nn.Linear(20736, 10000)
        self.fcl2 = nn.Linear(10000, NUM_ACTIONS)

    def forward(self, x):
        x =x.view(SIZE,SIZE)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.view(-1, 20736)
        x = F.relu(self.fcl1(x))
        x = self.fcl2(x)

        return x

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN(object):
    def __init__(self,turn,load=False,PATH1="model_offensive.pth",PATH2="model_defensive.pth",agent=AI):
        self.transitions = np.zeros((TRANSITION_CAP, 2 * NUM_STATES + 2))
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

    def select_action(self, x, G, turn, eps=0.1):
        G.all_valid(turn)
        available_pos = G.available
        if len(available_pos) == 0:
            return 64
        available_pos = list(map(lambda pos: G.size * pos[0] + pos[1], available_pos))

        if np.random.uniform() < eps:
            action = np.random.choice(available_pos, 1)[0]
        else:
            x = torch.as_tensor(x, dtype=torch.float).view(1, -1).to(device)
            action_v = self.Q(x)[0]
            ava_action = torch.as_tensor(action_v[available_pos])

            _, action_i = torch.max(ava_action, 0)

            action = available_pos[action_i]
        return action

    def store_transition(self, state, action, reward, succState):
        self.transitions[self.transitions_i % TRANSITION_CAP] = np.hstack((state, action, reward, succState))
        self.transitions_i += 1

    def learn(self, opponent):
        for step in range(10):
            if self.learn_iter % UPDATE_DELAY == 0:
                self.Q_.load_state_dict(self.Q.state_dict())
            self.learn_iter += 1

            sample_index = np.random.choice(TRANSITION_CAP, BATCH_SIZE)
            batch_trans = self.transitions[sample_index, :]

            batch_state = torch.as_tensor(batch_trans[:, :NUM_STATES], dtype=torch.float).to(device)
            batch_action = torch.as_tensor(batch_trans[:, NUM_STATES:NUM_STATES + 1], dtype=int).to(device)
            batch_reward = torch.as_tensor(batch_trans[:, NUM_STATES + 1:NUM_STATES + 2], dtype=torch.float).to(device)
            batch_succState = torch.as_tensor(batch_trans[:, NUM_STATES + 2:], dtype=torch.float).to(device)
            # print(batch_state)
            # print(batch_action)
            batch_y1 = self.Q(batch_state).gather(1,batch_action)
            batch_y2 = opponent(batch_succState).detach()
            # print(self.Q(batch_state))
            # print(batch_y1)
            batch_y2 = batch_reward - GAMMA * batch_y2.max(1)[0].view(-1, 1)
            # print(batch_y2)
            # print("-"*100)

            loss = F.smooth_l1_loss(batch_y1, batch_y2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

if __name__ == '__main__':
    offensive=DQN(player.BLACK,agent=Net1)
    defensive=DQN(player.WHITE,agent=Net1)

    for episode in range(EPISODE):
        G=game()
        round=0
        while True:
            round+=1
            state=G.state()
            act=offensive.select_action(state,G,player.BLACK)
            G.add_tensor(act,player.BLACK)
            # G.Display()
            reward=reward_dict[G.game_over()]
            succState=G.state()

            offensive.store_transition(state,act,reward,succState)

            if reward!=0 or round > 100:
                offensive.learn(defensive.Q)
                defensive.learn(offensive.Q)
                print('offensive Episode:{} | Reward:{}'.format(episode, reward))
                break

            state=G.state()
            act=defensive.select_action(state,G,player.WHITE)
            G.add_tensor(act,player.WHITE)
            reward = reward_dict[G.game_over()]
            succState = G.state()

            defensive.store_transition(state, act, reward, succState)

            if reward != 0:
                offensive.learn(defensive.Q)
                defensive.learn(offensive.Q)
                print('defensive Episode:{} | Reward:{}'.format(episode, reward))
                break

        if (episode + 1) % 100 == 0:
            torch.save(offensive.Q.state_dict(), 'model_offensive2.pth')
            torch.save(defensive.Q.state_dict(), 'model_defensive2.pth')
