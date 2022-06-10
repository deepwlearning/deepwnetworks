"""

The code for DQN is adapted from :
https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/405_DQN_Reinforcement_learning.py

"""

import random
import math
import numpy as np
from collections import namedtuple
# Importing our custom enviroment
from custom_envs.mountain_car.engine import MountainCar
# The libs we need to construct the DQN agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
# Import Replay Memory
from replay_buffer import ReplayBuffer

BATCH_SIZE = 1024
LR = 0.01                   # learning rate
EPSILON = 0.1              # starting epsilon for greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target DQN update frequency
MEMORY_SIZE = 10000         # size of the replay buffer

# Path to trained Net
PATH = 'torch_mountain_car_500episodes.pt'

# Gym Graphical Environment
GYM_SPEED = 1e8
GYM_GRAPH_STATE = False
GYM_RENDER = True
GYM_DEBUG = False
GYM_FRAME_STACK = 3

# Simulation Parameters
EPISODES = 500

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fp1 = nn.Linear(state_size, 128)
        self.fp2 = nn.Linear(128, 256)
        self.head_values = nn.Linear(256, 1)
        self.head_advantages = nn.Linear(256, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fp1(state))
        x = F.relu(self.fp2(x))
        values = self.head_values(x)
        advantages = self.head_advantages(x)
        return values + (advantages - advantages.mean())


class CNN(nn.Module):

    def __init__(self, h, w, outputs):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DWN(object):
    def __init__(self, input_shape, num_actions, w_learning=False, dnn_structure=True, batch_size=1024, epsilon=.25,
                 epsilon_decay=.9999, epsilon_min=.1, tau=.001, w_tau=0.001, memory_size=10000, learning_rate=.01,
                 gamma=.9, walpha=.001, per_epsilon=.001, beta_start=.4, beta_inc=1.002, seed=404):

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_states = self.input_shape[0]

        self.random_seed = seed

        # Learning parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.memory_size = memory_size

        self.beta = beta_start
        self.beta_inc = beta_inc

        # Construct NN models - Depending on the selected input
        self.dnn_structure = dnn_structure
        # Q-Network
        if self.dnn_structure:
            print("Using the DNN structure!")
            self.qnetwork_local = QNetwork(self.num_states, self.num_actions, self.random_seed).to(device)
            self.qnetwork_target = QNetwork(self.num_states, self.num_actions, self.random_seed).to(device)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        else:
            print("Using the CNN structure!")
            self.qnetwork_local = CNN(84, 84, self.num_actions).to(device)
            self.qnetwork_target = CNN(84, 84, self.num_actions).to(device)
            self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=self.learning_rate)
        # Replay memory
        self.memory = ReplayBuffer(self.num_actions, self.memory_size, self.batch_size, self.random_seed)
        self.per_epsilon = per_epsilon

        self.q_episode_loss = []

        # Init the values we need for W-learning
        self.w_learning = w_learning
        if self.w_learning:  # We only init the W-values if we need them, i.e., w_learning = True
            if self.dnn_structure:
                print("Using the DNN structure for W-learning!!!")
                self.wnetwork_local = QNetwork(self.num_states, 1, self.random_seed).to(device)
                self.wnetwork_target = QNetwork(self.num_states, 1, self.random_seed).to(device)
                self.optimizer_w = optim.Adam(self.wnetwork_local.parameters(), lr=self.learning_rate)
            else:
                print("Using the CNN structure for W-learning!!!")
                self.wnetwork_local = CNN(84, 84, 1).to(device)
                self.wnetwork_target = CNN(84, 84, 1).to(device)
                self.optimizer_w = optim.RMSprop(self.wnetwork_local.parameters(), lr=self.learning_rate)
            # Init the W net learning parameters and replay buffer
            self.memory_w = ReplayBuffer(self.num_actions, self.memory_size, self.batch_size, self.random_seed)
            self.w_alpha = walpha
            self.w_episode_loss = []
            self.w_tau = w_tau
            #self.wepsilon = wepsilon

    #
    # Chooses action based on epsilon-greedy policy
    #
    def choose_action(self, x):
        # Ensure state is tensor
        #print(type(x).__name__)
        if type(x).__name__ == 'ndarray':
            state = torch.from_numpy(x).float().unsqueeze(0).to(device)
        else:
            state = x
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.num_actions))

    #
    # Get W-value for the selected policy
    #
    def get_w_value(self, x):
        if type(x).__name__ == 'ndarray':
            state = torch.from_numpy(x).float().unsqueeze(0).to(device)
        else:
            state = x
        w_value = self.wnetwork_local.forward(state)
        w_value = w_value.detach()[0].data.numpy()
        w_value = w_value[0]
        return w_value

    #
    # Storing experience in the replay memory
    #
    def store_transition(self, s, a, r, s_, d):
        self.memory.add(s, a, r, s_, d)

    #
    # Storing experience in the replay memory for the W-learning part
    #
    def store_w_transition(self, s, a, r, s_, d):
        self.memory_w.add(s, a, r, s_, d)

    #
    # Trains the Q model using experiences randomly sampled from the replay memory
    #
    def learn(self):
        if len(self.memory) > self.batch_size:
            states, actions, rewards, next_states, probabilites, experiences_idx, dones = self.memory.sample()

            current_qs = self.qnetwork_local(states).gather(1, actions)
            next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            max_next_qs = self.qnetwork_target(next_states).detach().gather(1, next_actions)
            target_qs = rewards + self.gamma * max_next_qs

            is_weights = np.power(probabilites * self.batch_size, -self.beta)
            is_weights = torch.from_numpy(is_weights / np.max(is_weights)).float().to(device)
            loss = (target_qs - current_qs).pow(2) * is_weights
            loss = loss.mean()
            # To track the loss over episode
            self.q_episode_loss.append(loss.detach().numpy())

            td_errors = (target_qs - current_qs).detach().numpy()
            self.memory.update_priorities(experiences_idx, td_errors, self.per_epsilon)

            self.qnetwork_local.zero_grad()
            loss.backward()
            self.optimizer.step()
            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
            # ------------------ update exploration rate ------------------ #
            self.epsilon = max(self.epsilon_decay * self.epsilon, self.epsilon_min)

    #
    # Trains the W model using experiences randomly sampled from the W replay memory
    #
    def learn_w(self):
        # W target parameter update
        if len(self.memory_w) > self.batch_size:
            states, actions, rewards, next_states, probabilites, experiences_idx, dones = self.memory_w.sample()

            # Calculate the Q-values as in normal Q-learning
            current_qs = self.qnetwork_local(states).gather(1, actions)
            next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            max_next_qs = self.qnetwork_target(next_states).detach().gather(1, next_actions)
            target_qs = rewards + self.gamma * max_next_qs

            # Calculate the W-values, as proposed with Eq. (3) in "W-learning Competition among selfish Q-learners"
            current_w = self.wnetwork_local(states).detach()
            target_w = (1 - self.w_alpha) * current_w + self.w_alpha * (current_qs - target_qs)

            is_weights = np.power(probabilites * self.batch_size, - self.beta)
            is_weights = torch.from_numpy(is_weights / np.max(is_weights)).float().to(device)
            w_loss = (target_w - current_w).pow(2) * is_weights
            w_loss = w_loss.mean()
            # To track the loss over episode
            self.w_episode_loss.append(w_loss.detach().numpy())

            td_errors = (target_qs - current_qs).detach().numpy()
            self.memory_w.update_priorities(experiences_idx, td_errors, self.per_epsilon)

            self.wnetwork_local.zero_grad()
            w_loss.backward()
            self.optimizer_w.step()
            # ------------------- update target network ------------------- #
            self.soft_update(self.wnetwork_local, self.wnetwork_target, self.w_tau)

    #
    # Soft update of the target neural network
    #
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    #
    # Update parameters after the episode
    #
    def update_params(self):
        #self.eps = max(self.eps_end, self.eps_decay * self.eps)  # TO DO: rename them to the correct format
        self.beta = min(1.0, self.beta_inc * self.beta)

    #
    # Saves parameters of a trained Q model
    #
    def save_net(self, path):
        torch.save(self.qnetwork_local.state_dict(), path)

    #
    # Saves parameters of a trained W model
    #
    def save_w_net(self, path):
        torch.save(self.wnetwork_local.state_dict(), path)

    #
    # Loads a saved Q model
    #
    def load_net(self, path):
        self.qnetwork_local.load_state_dict(torch.load(path)), self.qnetwork_target.load_state_dict(torch.load(path))
        self.qnetwork_local.eval(), self.qnetwork_target.eval()

    #
    # Loads a saved W model
    #
    def load_w_net(self, path):
        if self.w_learning:
            self.wnetwork_local.load_state_dict(torch.load(path)), self.wnetwork_target.load_state_dict(
                torch.load(path))
            self.wnetwork_local.eval(), self.wnetwork_target.eval()

    #
    # Collect the loss of Q and W learning
    #
    def collect_loss_info(self):
        #print("Q-episode loss:", self.q_episode_loss)
        avg_q_loss = np.average(self.q_episode_loss)
        avg_w_loss = 0
        self.q_episode_loss = []
        if self.w_learning:
            #print("W-episode loss:", self.w_episode_loss)
            avg_w_loss = np.average(self.w_episode_loss)
            self.w_episode_loss = []
        return avg_q_loss, avg_w_loss

