import random
import math
import numpy as np
import collections
# Importing our custom enviroment
from custom_envs.mountain_car.engine import MountainCar
from dqn_agent import DWN

BATCH_SIZE = 1024
LR = 0.01                   # learning rate
EPSILON = 0.25              # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target DQN update frequency
MEMORY_SIZE = 10000         # size of the replay buffer

# Path to trained Net, without extensions as used in the per net...
PATH = 'saved_neural_networks/torch_mountain_car_'

# Gym Graphical Environment
GYM_SPEED = 1e8
GYM_GRAPH_STATE = False
GYM_RENDER = True
GYM_DEBUG = False
GYM_FRAME_STACK = 3

# Simulation Parameters
EPISODES = 10


class DWL(object):
    def __init__(self, input_shape, num_actions, num_policies, w_learning=True, dnn_structure=True, batch_size=1024,
                 epsilon=.25, epsilon_decay=.995, epsilon_min=.1, wepsilon=.99, wepsilon_decay=.9995, wepsilon_min=.1,
                 memory_size=10000, target_replace_fre=1000, learning_rate=.01, gamma=.9):

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_policies = num_policies  # The number of policies is the number of
        self.num_states = self.input_shape[0]
        self.dnn_structure = dnn_structure
        # Selecting type of Neural network, DNN or CNN
        # self.ccn_ann = cnn_ann

        # Learning parameters for DQN agents
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.target_replace_fre = target_replace_fre

        # Learning parameters for W learning net
        self.w_learning = w_learning
        self.wepsilon = wepsilon
        self.wepsilon_decay = wepsilon_decay
        self.wepsilon_min = wepsilon_min
        # Construct Agents for each policy
        self.agents = []
        for i in range(self.num_policies):
            self.agents.append(DWN(self.input_shape, self.num_actions, dnn_structure=self.dnn_structure,
                                   batch_size=self.batch_size, epsilon=self.epsilon,
                                   epsilon_decay=self.epsilon_decay, epsilon_min=self.epsilon_min,
                                   w_learning=self.w_learning, memory_size=self.memory_size,
                                   learning_rate=self.learning_rate, gamma=self.gamma))

    #
    # Action nomination function
    #
    def get_action_nomination(self, x):
        nominated_actions = []
        w_values = []
        for agent in self.agents:
            nominated_actions.append(agent.choose_action(x))
            w_values.append(agent.get_w_value(x))

        # Try different W-policies the same logic as exploration vs explotation
        if np.random.uniform() > self.wepsilon:
            policy_sel = np.argmax(w_values)
        else:
            policy_sel = np.random.randint(0, self.num_policies)
            self.wepsilon = max(self.wepsilon * self.wepsilon_decay, self.wepsilon_min)

        sel_action = nominated_actions[policy_sel]
        return sel_action, policy_sel

    #
    # Store experiences to all agents
    #
    def store_transition(self, s, a, rewards, s_, d, policy_sel):
        for i in range(self.num_policies):
            self.agents[i].store_transition(s, a, rewards[i], s_, d)
            if i != policy_sel:  # Do not store experience of the policy we selected
                self.agents[i].store_w_transition(s, a, rewards[i], s_, d)

    #
    # Get loss values for Q and W
    #
    def get_loss_values(self):
        q_loss, w_loss, = [], []
        for i in range(self.num_policies):
            q_loss_part, w_loss_part = self.agents[i].collect_loss_info()
            q_loss.append(q_loss_part), w_loss.append(w_loss_part)
        #print(q_loss, w_loss)
        return q_loss, w_loss

    #
    # Train Q and W networks
    #
    def learn(self):
        for i in range(self.num_policies):
            self.agents[i].learn()
            #if self.init_learn_steps_count > self.init_learn_steps_num: # we start training W-network with delay
            self.agents[i].learn_w()

    #
    # Updating Beta and exploration rate
    #
    def update_params(self):
        for i in range(self.num_policies):
            self.agents[i].update_params()

    #
    # Save trained Q-networks and W-networks to file
    #
    def save(self, path):
        for i in range(self.num_policies):
            self.agents[i].save_net(path + 'Q' + str(i) + '.pt')
            self.agents[i].save_w_net(path + 'W' + str(i) + '.pt')

    #
    # Load the pre-trained Q-networks and W-networks from file
    #
    def load(self, path):
        for i in range(self.num_policies):
            print("Loading", i)
            self.agents[i].load_net(path + 'Q' + str(i) + '.pt')
            self.agents[i].load_w_net(path + 'W' + str(i) + '.pt')

