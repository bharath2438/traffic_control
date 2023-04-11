import torch
import random
import numpy as np
from collections import deque
# from game import SnakeGameAI, Direction, Point
from model import DQN, ResNet, ResidualBlock, QTrainer
from model_generator import ModelGenerator
import json
import matplotlib
import matplotlib.pyplot as plt
import cv2
import warnings
from helper import plot
import time

warnings.filterwarnings("ignore")

MAX_MEMORY = 50000
BATCH_SIZE = 100
LR = 0.001

def train():
##    plot_scores = []
##    plot_mean_scores = []
##    total_score = 0
##    record = 0
    plot_waiting_times = []
    agent = Agent()
    use_cuda = torch.cuda.is_available()

    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    agent.Tensor = FloatTensor

    agent.is_ipython = 'inline' in matplotlib.get_backend()
    plt.ion()
    
    if use_cuda:
        print("training on GPU")
        agent.model.cuda()
    else:
        print("cpu")
    done = 0
    model_save = 0
    while True:
        # get old state
        state_old = agent.get_state()
        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        agent.send_action(json.dumps({"actions":[{"junctionId":"1", "action":final_move}]}))
        reward = agent.receive_rewards()
        state_new = agent.get_state()        

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new)

        # remember
        agent.remember(state_old, final_move, reward, state_new)

        if done == 20:
            # train long memory, plot result
            agent.n_games += 1
            agent.train_long_memory()
            done = 0
            plot_waiting_times.append(reward)
            plot(plot_waiting_times)

        if model_save % 100 == 0:
            torch.save(agent.model.state_dict(), 'F://traffic3d//backend//models//model_'+str(model_save))

        done += 1
        model_save += 1

def test():
    agent = Agent()
    use_cuda = torch.cuda.is_available()

    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    agent.Tensor = FloatTensor

    agent.is_ipython = 'inline' in matplotlib.get_backend()
    plt.ion()
    
    if use_cuda:
        agent.model.cuda()
        
    agent.model.load_state_dict(torch.load('F://traffic3d//backend//models//model_600'))
    agent.model.eval()

    timestep = 0

    while True:

        if timestep == 25:
            state_cur = agent.get_state()
            move = agent.get_action(state_cur, use_epsilon=False)
            agent.send_action(json.dumps({"actions":[{"junctionId":"1", "action":move}]}))
            reward = agent.receive_rewards()
            state_new = agent.get_state()
            timestep = 0
        timestep += 1
    

class Agent(ModelGenerator):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        #self.model = ResNet(ResidualBlock, [2, 2, 2, 2], 4)
        self.model = DQN(4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.setup_socket()

    def get_state(self):
        img = self.receive_images()["1"]
        bg = cv2.imread("C://Users/win10/Desktop/test.png")
        img = cv2.subtract(img, bg)
        img = self.prepro(img)
        return img        

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states = zip(*mini_sample)        
        
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)

        print(states.shape)
        
        self.trainer.train_step(states, actions, rewards, next_states)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            
        self.trainer.train_step(state, action, reward, next_state)

    def get_action(self, state, use_epsilon=True):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 150 - self.n_games
        if random.randint(0, 200) < self.epsilon and use_epsilon:
            #print("exploring")
            move = random.randint(0, 3)
            final_move = move
        else:
            #print("exploiting")
            probs = self.model(torch.autograd.Variable(state)) #torch.autograd.Variable(state)
            m = torch.distributions.Categorical(probs)            
            move = m.sample()
            final_move = move.data[0].tolist()
        print(final_move)
        return final_move

train()
#test()
