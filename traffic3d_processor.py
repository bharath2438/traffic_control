import model_generator

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
import torch
from torch import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
    

class Traffic3DProcessor(model_generator.ModelGenerator):
    def __init__(self):
        super().__init__()

    def enable(self):
        # Using CUDA GPU
        use_cuda = torch.cuda.is_available()

        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.Tensor = FloatTensor

        self.is_ipython = 'inline' in matplotlib.get_backend()
        plt.ion()

        PATHmodel2 = 'emerMFD2.pt'

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.target = DQN(self.max_number_of_junction_states).to(device)
        self.target.load_state_dict(torch.load(PATHmodel2).state_dict())
        self.target.eval()
        
        # hyperparameters
        self.gamma = 0.999
        # self.policy = DQN(self.max_number_of_junction_states)
        self.policy = self.target
        if use_cuda:
            self.policy.cuda()
##        optimizer = optim.RMSprop(self.policy.parameters(), lr=0.001)
##        eps = np.finfo(np.float32).eps
##        for u in torch.arange(1, 11):
##            for v in torch.arange(1, 11):
##                self.run()
##            optimizer.step()
##            print("optim step up")
##            torch.save(self.policy, PATHmodel2)
##            optimizer.zero_grad()
##            print("gradients zeroed")
##            del self.policy.basic_rewards[:]
##            del self.policy.saved_log_probs[:]
##            print("things deleted")
        while True:
            self.run()

    def run(self):
        R = 0
        policy_loss = []
        discounted_rewards = []
        #eps = np.finfo(np.float32).eps
        eps = 0.25
        # One episode is made of 100 timesteps, get 100 rewards after an episode.
        for i in torch.arange(1, 11):
            imgs = self.receive_images()
            actions = {"actions": []}
            for junction_id in imgs:
                img1 = self.prepro(imgs[junction_id])
                actions["actions"].append({"junctionId": junction_id, "action": self.select_action(img1)})
            self.send_action(json.dumps(actions))
            rew = self.receive_rewards()
            self.policy.basic_rewards.append(rew)

##        for r in self.policy.basic_rewards[::-1]:
##            R = r + self.gamma * R
##            discounted_rewards.insert(0, R)
##        discounted_rewards = torch.FloatTensor(discounted_rewards)
##        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
##                discounted_rewards.std() + eps)
##        for log_prob, reward in zip(self.policy.saved_log_probs, discounted_rewards):
##            policy_loss.append(-log_prob * reward)
##        policy_loss = torch.cat(policy_loss).sum()
##        # As long as you have retain_raph=True, you can do backprop at any time.
##        policy_loss.backward(retain_graph=True)
##        print("done backprop")

    def select_action(self, state):
        probs1 = self.policy(Variable(state))
        m = Categorical(probs1)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
#        print(action.data[0].tolist())
        print(action)
        return action.data[0].tolist()

    def prepro(self, x):
        x = cv2.resize(x, (100, 100))
        x = x.transpose(2, 0, 1)
        x = np.ascontiguousarray(x, dtype=np.float32) / 255
        x = torch.from_numpy(x)
        return (x.unsqueeze(0).type(self.Tensor))


class DQN(nn.Module):
    def __init__(self, max_number_of_junction_states):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.head = nn.Linear(576, max_number_of_junction_states)

        self.saved_log_probs = []
        self.basic_rewards = []

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.head(x.view(x.size(0), -1))
        return F.softmax(x, dim=1)


Traffic3DProcessor()
