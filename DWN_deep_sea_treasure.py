import numpy as np
import pandas as pd
import collections
from mo_envs.mo_deep_sea_treasure_env import MODeepSeaTresureEnv
from PIL import Image

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

# Import deep W-learning agent
from dqw_agent import DWL

TREASURE_VALUES = [-1, -5, +1, +2, +3, +5, +8, +16, +24, +50, +74, +124]
EPISODES = 500

# The Q-learning agent parameters
BATCH_SIZE = 1024           # Batch size
LR = 0.01                   # learning rate
EPSILON = .25  # .95               # starting epsilon for greedy policy
EPSILON_MIN = .25           # the minimal epsilon we want
EPSILON_DECAY = .99995      # epsilon decay
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target DQN update frequency
MEMORY_SIZE = 100000        # size of the replay buffer

# The W-learning parameters
WEPSILON = 0.01
WEPSILON_DECAY = 0.9995
WEPSILON_MIN = 0.01

# Path to trained Net
PATH = 'saved_nets/deep_sea_wlearning'

resize = T.Compose([T.ToPILImage(),
                    T.Resize(84, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_screen(environment):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = environment.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


if __name__ == "__main__":
    # Import the Deep Sea Treasure Environment from
    env = MODeepSeaTresureEnv(from_pixels=True)
    state = env.reset()
    init_screen = get_screen(env)
    _1, _2, screen_height, screen_width = init_screen.shape
    num_objectives = 2  # We have two objections, moving and end reward
    print("Screen information:", _1, _2, screen_height, screen_width)
    num_actions = env.n_actions
    print("Information about environment, state shape:", state.shape, " action space:", num_actions,
          "number of objective:", num_objectives)
    # Init the W-learning agent
    agent = DWL(state.shape, num_actions, num_objectives, dnn_structure=False, epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                epsilon_decay=EPSILON_DECAY, wepsilon=WEPSILON, wepsilon_decay=WEPSILON_DECAY,
                wepsilon_min=WEPSILON_MIN, memory_size=MEMORY_SIZE, learning_rate=LR, gamma=GAMMA)
    # Init list for information we need to collect during simulation
    agent.load(PATH)
    num_of_steps = []
    # With 2 objectives
    coll_reward1 = []
    coll_reward2 = []
    loss_q1_episode = []
    loss_q2_episode = []
    loss_w1_episode = []
    loss_w2_episode = []
    pol1_sel_episode = []
    pol2_sel_episode = []
    for episode in range(EPISODES):
        done = False
        env.reset()
        state = get_screen(env)
        rewardsSum1 = 0
        rewardsSum2 = 0
        qSum = 0
        qActions = 1
        lossSum = 0

        num_steps = 0
        selected_policies = []
        while not done:
            action, sel_policy = agent.get_action_nomination(state)
            nextState, r, done, _ = env.step(action)
            nextState = get_screen(env)
            reward = [np.sum(r[:2] * TREASURE_VALUES[:2]), np.sum(r[2:] * TREASURE_VALUES[2:])]
            agent.store_transition(state, action, reward, nextState, done, sel_policy)
            selected_policies.append(sel_policy)
            env.render()
            #agent.learn()
            state = nextState
            num_steps += 1

            rewardsSum1 = np.add(rewardsSum1, reward[0])
            rewardsSum2 = np.add(rewardsSum2, reward[1])
            state = nextState
        # End of the episode
        q_loss, w_loss = agent.get_loss_values()
        print("Episode", episode, "end_reward", reward, "Sum of the reward:", qSum, "Num steps:", num_steps,
              "Epsilon:", agent.epsilon, "Q loss:", q_loss, "W loss", w_loss)
        count_policies = collections.Counter(selected_policies)
        print("Policies selected in the episode:", count_policies, "Policy 1:", count_policies[0],
              "Policy 2:", count_policies[1])
        count_policies = collections.Counter(selected_policies)
        # Save the performance to lists
        num_of_steps.append(num_steps)
        coll_reward1.append(rewardsSum1)
        coll_reward2.append(rewardsSum2)
        pol1_sel_episode.append(count_policies[0])
        pol2_sel_episode.append(count_policies[1])
        loss_q1_episode.append(q_loss[0])
        loss_q2_episode.append(q_loss[1])
        loss_w1_episode.append(w_loss[0])
        loss_w2_episode.append(w_loss[1])

        agent.update_params()
    # Save the results
    df_results = pd.DataFrame()
    df_results['episodes'] = range(1, EPISODES + 1)
    df_results['num_steps'] = num_of_steps
    df_results['col_reward1'] = coll_reward1
    df_results['col_reward2'] = coll_reward2
    df_results['policy1'] = pol1_sel_episode
    df_results['policy2'] = pol2_sel_episode
    df_results['loss_q1'] = loss_q1_episode
    df_results['loss_q2'] = loss_q2_episode
    df_results['loss_w1'] = loss_w1_episode
    df_results['loss_w2'] = loss_w2_episode
    df_results.to_csv('results/DWL_deep_sea_exploit_test.csv')
    # Save the trained ANN - uncomment if you need to save ANN
    # agent.save(PATH)

