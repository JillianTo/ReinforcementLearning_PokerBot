##
## Adapted from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
##

import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, internal_features=128):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, internal_features)
        self.layer2 = nn.Linear(internal_features, internal_features)
        self.layer3 = nn.Linear(internal_features, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
