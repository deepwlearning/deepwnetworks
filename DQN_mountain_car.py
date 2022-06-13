import pandas as pd
import numpy as np
from dqn_agent import DWN
from custom_envs.mountain_car.engine import MountainCar
import torch

# Gym Graphical Environment
GYM_SPEED = 1e8
GYM_GRAPH_STATE = False
GYM_RENDER = True
GYM_DEBUG = False
GYM_FRAME_STACK = 3

# The Q-learning agent parameters
BATCH_SIZE = 1024
LR = .01                   # learning rate
EPSILON = .01  # .95               # starting epsilon for greedy policy
EPSILON_MIN = .01           # The minimal epsilon we want
EPSILON_DECAY = .99995      # The minimal epsilon we want
GAMMA = .99                # reward discount
MEMORY_SIZE = 10000         # size of the replay buffer


# Simulation Parameters
EPISODES = 2500

# Path to trained Net
PATH = 'saved_nets/torch_mountain_car.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # IMPORT A MULTI-OBJECTIVE gym enviroment
    env = MountainCar(speed=GYM_SPEED, graphical_state=GYM_GRAPH_STATE, render=GYM_RENDER, is_debug=GYM_DEBUG,
                      frame_stack=GYM_FRAME_STACK)
    # Get environment information
    state = env.get_state()
    state_shape = state.shape
    action_num = env.get_num_of_actions()
    print("Information about environment, state shape:", state_shape, " action space:", action_num)
    # Init the DQN agent
    agent = DWN(state_shape, action_num, batch_size=BATCH_SIZE, epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                epsilon_decay=EPSILON_DECAY, memory_size=MEMORY_SIZE, learning_rate=LR, gamma=GAMMA)
    agent.load_net(PATH)
    # Init list for information we need to collect during simulation
    num_of_steps = []
    coll_rewards = []
    loss_episode = []

    for episode in range(EPISODES):
        done = False
        state = env.reset()
        rewardsSum = 0
        num_steps = 0

        while not done:
            action = agent.choose_action(state)
            num_steps += 1
            nextState, reward, done, _ = env.step_all(action)
            if done and nextState[0] > 0.5:
                reward = 100
            else:
                reward = -1
            rewardsSum = np.add(rewardsSum, reward)
            nextState = nextState
            agent.store_transition(state, action, reward, nextState, done)
            agent.learn()
            state = nextState
        print("Episode", episode, "end_reward", reward, "Sum of the reward:", rewardsSum, "Num steps:", num_steps,
              "Epsilon:", agent.epsilon)
        # Save the required information
        num_of_steps.append(num_steps)
        coll_rewards.append(rewardsSum)
        loss_episode.append(np.round(agent.collect_loss_info()[0], 8))
        # Update the parameters
        agent.update_params()

    # Save the trained ANN
    agent.save_net(PATH)

