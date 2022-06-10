import pandas as pd
import numpy as np
import math
import collections
from dqw_agent import DWL
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
EPSILON = .05  # .95               # starting epsilon for greedy policy
EPSILON_MIN = .1           # The minimal epsilon we want
EPSILON_DECAY = .99995      # The minimal epsilon we want
GAMMA = .99                # reward discount
MEMORY_SIZE = 10000         # size of the replay buffer

# The W-learning parameters
WEPSILON = 0.01  #.99
WEPSILON_DECAY = 0.9995
WEPSILON_MIN = 0.01

# Simulation Parameters
EPISODES = 2500

# Path to trained net
PATH = 'saved_nets/torch_mountain_car_wlearning'

if __name__ == "__main__":
    # IMPORT A MULTI-OBJECTIVE gym enviroment
    env = MountainCar(speed=GYM_SPEED, graphical_state=GYM_GRAPH_STATE, render=GYM_RENDER, is_debug=GYM_DEBUG,
                      frame_stack=GYM_FRAME_STACK)

    # Get the environment information we need
    state = env.get_state()
    state_shape = state.shape
    action_num = env.get_num_of_actions()
    num_objectives = env.get_num_of_objectives()
    print("Information about environment, state shape:", state_shape, " action space:", action_num,
          "number of objective:", num_objectives)
    # Init the W-learning agent
    agent = DWL(state_shape, action_num, num_objectives, dnn_structure=True, epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                epsilon_decay=EPSILON_DECAY, wepsilon=WEPSILON, wepsilon_decay=WEPSILON_DECAY,
                wepsilon_min=WEPSILON_MIN, memory_size=MEMORY_SIZE, learning_rate=LR, gamma=GAMMA)
    # Init list for information we need to collect during simulation
    num_of_steps = []
    agent.load(PATH)
    # With 3 objectives
    coll_reward1 = []
    coll_reward2 = []
    coll_reward3 = []
    loss_q1_episode = []
    loss_q2_episode = []
    loss_q3_episode = []
    loss_w1_episode = []
    loss_w2_episode = []
    loss_w3_episode = []
    pol1_sel_episode = []
    pol2_sel_episode = []
    pol3_sel_episode = []

    for episode in range(EPISODES):
        done = False
        rewardsSum1 = 0
        rewardsSum2 = 0
        rewardsSum3 = 0
        qSum = 0
        qActions = 1
        lossSum = 0
        state = env.reset()

        num_steps = 0
        selected_policies = []
        while not done:
            nom_action, sel_policy = agent.get_action_nomination(state)
            num_steps += 1
            nextState, reward, done, _ = env.step_all(nom_action)
            selected_policies.append(sel_policy)
            nextState = nextState
            # The goal reward is it not given correctly by the reward, we also increased it
            if done and nextState[0] > 0.5:
                reward = [100, 100, 0]  # If it is the final reward
            agent.store_transition(state, nom_action, reward, nextState, done, sel_policy)
            agent.learn()
            rewardsSum1 = np.add(rewardsSum1, reward[0])
            rewardsSum2 = np.add(rewardsSum2, reward[1])
            rewardsSum3 = np.add(rewardsSum3, reward[2])
            state = nextState
        # End of the episode
        q_loss, w_loss = agent.get_loss_values()
        print("Episode", episode, "end_reward", reward, "Sum of the reward:", qSum, "Num steps:", num_steps,
              "Epsilon:", agent.epsilon, "Q loss:", q_loss, "W loss", w_loss)
        count_policies = collections.Counter(selected_policies)
        print("Policies selected in the episode:", count_policies, "Policy 1:", count_policies[0],
              "Policy 2:", count_policies[1], "Policy 3:", count_policies[2])
        count_policies = collections.Counter(selected_policies)
        #q_losses, w_losses = agent.get_loss_values()
        # Save the performance to lists
        num_of_steps.append(num_steps)
        coll_reward1.append(rewardsSum1)
        coll_reward2.append(rewardsSum2)
        coll_reward3.append(rewardsSum3)
        pol1_sel_episode.append(count_policies[0])
        pol2_sel_episode.append(count_policies[1])
        pol3_sel_episode.append(count_policies[2])
        loss_q1_episode.append(q_loss[0])
        loss_q2_episode.append(q_loss[1])
        loss_q3_episode.append(q_loss[2])
        loss_w1_episode.append(w_loss[0])
        loss_w2_episode.append(w_loss[1])
        loss_w3_episode.append(w_loss[2])

        agent.update_params()

    # Save the results
    df_results = pd.DataFrame()
    df_results['episodes'] = range(1, EPISODES + 1)
    df_results['num_steps'] = num_of_steps
    df_results['col_reward1'] = coll_reward1
    df_results['col_reward2'] = coll_reward2
    df_results['col_reward3'] = coll_reward3
    df_results['policy1'] = pol1_sel_episode
    df_results['policy2'] = pol2_sel_episode
    df_results['policy3'] = pol3_sel_episode
    df_results['loss_q1'] = loss_q1_episode
    df_results['loss_q2'] = loss_q2_episode
    df_results['loss_q3'] = loss_q3_episode
    df_results['loss_w1'] = loss_w1_episode
    df_results['loss_w2'] = loss_w2_episode
    df_results['loss_w3'] = loss_w3_episode
    #df_results.to_csv('results/DWL_mountain_car.csv')
    # Save the trained ANN
    #agent.save(PATH)

