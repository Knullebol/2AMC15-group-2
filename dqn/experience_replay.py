from collections import deque, namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self, max_length, seed = None):
        self.memory = deque([], maxlen = max_length)

        if(seed is not None):
            random.seed(seed)

    def append(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)