from reversi import *
import torch,math
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable
from collections import Counter
from torch.distributions import Categorical

STATE_CNT=SIZE*SIZE
ACTION_CNT=SIZE*SIZE+1

TRANSITION_CAP=200
UPDATE_DELAY=100
BATCH_SIZE=32
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

pi = Variable(torch.FloatTensor([math.pi])).cuda()


def normal(x, mu, sigma_sq):
    a = (-1 * (Variable(x) - mu).pow(2) / (2 * sigma_sq)).exp()
    b = 1 / (2 * sigma_sq * pi.expand_as(sigma_sq)).sqrt()
    return a * b


class Policy(nn.Module):
    def __init__(self, hidden_size=128):
        super(Policy, self).__init__()

        self.linear1 = nn.Linear(STATE_CNT, hidden_size)
        self.linear2 = nn.Linear(hidden_size, ACTION_CNT)
        self.linear2_ = nn.Linear(hidden_size, ACTION_CNT)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma_sq = self.linear2_(x)

        return mu, sigma_sq


class REINFORCE:
    def __init__(self,G,turn):
        self.model1,self.model2 = Policy(),Policy()
        self.model1 = self.model1.cuda()
        self.model2 = self.model1.cuda()
        self.optimizer = optim.Adam(self.model1.parameters(), lr=1e-3)
        self.model1.train()
        self.game=G
        self.turn=turn
    def select_action(self, state):
        self.game.all_valid(self.turn)
        available_pos = list(map(lambda pos: G.size * pos[0] + pos[1], G.available))

        mu, sigma_sq = self.model1(Variable(state).cuda())
        sigma_sq = F.softplus(sigma_sq)

        eps = torch.randn(mu.size())
        # calculate the probability
        action = (mu + sigma_sq.sqrt() * Variable(eps).cuda()).data
        prob = normal(action, mu, sigma_sq)
        entropy = -0.5 * ((sigma_sq + 2 * pi.expand_as(sigma_sq)).log() + 1)
        av_prob=prob[available_pos]
        av_prob/=torch.sum(av_prob)

        m = Categorical(av_prob).sample()
        if type(m)==type(1):
            action = available_pos[m.item()]
        else:
            action=available_pos[m.item()[0]]
        log_prob = prob.log()
        return action, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss = loss - (log_probs[i] * (Variable(R).expand_as(log_probs[i])).cuda()).sum() - (0.0001 * entropies[i].cuda()).sum()
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.model1.parameters(), 40)
        self.optimizer.step()

if __name__ == '__main__':
    G=game()
    offensive=REINFORCE(G,player.BLACK)
    defensive=REINFORCE(G,player.WHITE)
    for i_episode in range(EPISODE):
        state = torch.Tensor(G.state())
        entropies = []
        log_probs = []
        rewards = []
        for t in range(10):
            action, log_prob, entropy = offensive.select_action(state)
            G.add_tensor(action,player.BLACK)
            reward=reward_dict[G.game_over()]
            done = G.game_over()!=0
            next_state=G.state()

            entropies.append(entropy)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = torch.Tensor([next_state])

            if done:
                break

        offensive.update_parameters(rewards, log_probs, entropies, GAMMA)

        state = torch.Tensor(G.state())
        entropies = []
        log_probs = []
        rewards = []
        for t in range(10):
            action, log_prob, entropy = defensive.select_action(state)
            G.add_tensor(action,player.BLACK)
            reward=reward_dict[G.game_over()]
            done = G.game_over()!=0
            next_state=G.state()

            entropies.append(entropy)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = torch.Tensor([next_state])

            if done:
                break

        defensive.update_parameters(rewards, log_probs, entropies, GAMMA)

        if i_episode % 100 == 0:
            torch.save(offensive.model1.state_dict(), 'model_offensive.pkl')

        print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
