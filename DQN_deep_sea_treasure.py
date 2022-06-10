import numpy as np
import pandas as pd
from mo_envs.mo_deep_sea_treasure_env import MODeepSeaTresureEnv
from PIL import Image

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

#from un_used_material.dqn import DQNAgent
from dqn_agent import DWN

TREASURE_VALUES = [-1, -5, +1, +2, +3, +5, +8, +16, +24, +50, +74, +124]
EPISODES = 500

# The Q-learning agent parameters
BATCH_SIZE = 1024           # Batch size
LR = 0.01                   # learning rate
EPSILON = .01 #95               # starting epsilon for greedy policy
EPSILON_MIN = .01 #25           # the minimal epsilon we want
EPSILON_DECAY = .99995      # epsilon decay
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target DQN update frequency
MEMORY_SIZE = 100000        # size of the replay buffer

# Path to trained Net
PATH = 'saved_nets/deep_sea_basic.pt'

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
    print("Screen information:", _1, _2, screen_height, screen_width)
    num_actions = env.n_actions
    print("Information about environment, state shape:", state.shape, " action space:", num_actions)
    agent = DWN(state.shape, num_actions, dnn_structure=False, batch_size=BATCH_SIZE, epsilon=EPSILON,
                epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN, memory_size=MEMORY_SIZE, learning_rate=LR,
                gamma=GAMMA)
    agent.load_net(PATH)
    # Init list for information we need to collect during simulation
    num_of_steps = []
    coll_rewards = []
    loss_episode = []

    for episode in range(EPISODES):
        done = False
        env.reset()
        state = get_screen(env)
        num_steps = 0
        rewardsSum = 0
        while not done:
            action = agent.choose_action(state)
            nextState, r, done, _ = env.step(action)
            nextState = get_screen(env)
            reward = sum(r*TREASURE_VALUES)
            rewardsSum = np.add(rewardsSum, reward)
            agent.store_transition(state, action, reward, nextState, done)
            env.render()
            #agent.learn()
            state = nextState
            num_steps += 1
        print("Episode", episode, "Number of moves in the episode", num_steps, "end_reward", reward, "exploration rate",
              "Epsilon:", agent.epsilon)
        # Save the required information
        num_of_steps.append(num_steps)
        coll_rewards.append(rewardsSum)
        loss_episode.append(np.round(agent.collect_loss_info()[0], 8))
        # Update the beta parameters
        agent.update_params()  # Update the beta values
    # Save the results
    df_results = pd.DataFrame()
    df_results['episodes'] = range(1, EPISODES + 1)
    df_results['num_steps'] = num_of_steps
    df_results['col_reward'] = coll_rewards
    df_results['loss_episo'] = loss_episode
    df_results.to_csv('results/DQN_deep_sea_greedy.csv')
    # Save the trained ANN
    #agent.save_net(PATH)

