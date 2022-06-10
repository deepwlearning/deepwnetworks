import numpy as np
import random
from collections import namedtuple, deque
import torch

SMOOTH_SAMPLING = 0.6             # This is alpha factor as per original paper "Prioritised Experience Replay"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.priorities = np.zeros(buffer_size)
        # self.seed = random.seed(seed)
        self.max_priority = 1

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        if len(self.memory) == self.buffer_size:
            np.roll(self.priorities, -1)
            self.priorities[-1] = self.max_priority
        else:
            self.priorities[max(len(self.memory) - 1, 0)] = self.max_priority

    def update_priorities(self, experiences_idx, td_errors, epsilon):
        self.priorities[experiences_idx] = np.abs(td_errors).reshape(-1) + epsilon
        #print("Priorities values:", np.abs(td_errors).reshape(-1))
        #print("Priorities values after we add epsilon:", np.abs(td_errors).reshape(-1) + epsilon)
        self.max_priority = np.max(self.priorities)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        probabilities = self.probs(self.priorities)[:len(self.memory)]
        experiences_idx = np.random.choice(len(self.memory), size=self.batch_size, p=probabilities)
        experiences = [self.memory[idx] for idx in experiences_idx]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        probabilities = probabilities[experiences_idx]
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, probabilities, experiences_idx, dones

    def probs(self, priorities):
        probabilities = np.power(np.array(priorities), SMOOTH_SAMPLING)
        return probabilities / np.sum(probabilities)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

