# custom_model.py

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

import torch.nn as nn

import isaacgym #BugFix
import torch

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, hidden_size=256, 
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        print("Policy: observation_space", observation_space)
        print("Policy: action_space", action_space)
        print("Policy: num_observations", self.num_observations)
        self.net = nn.Sequential(nn.Linear(self.num_observations, hidden_size),
                                 nn.ReLU(), #Also Tanh() or ELU()
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(), #Also Tanh() or ELU()
                                 nn.Linear(hidden_size, hidden_size // 2),
                                 nn.ReLU()  #Also Tanh() or ELU()
                                 )
        self.mean_layer = nn.Linear(hidden_size // 2, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
    
    def act(self, inputs, role):
        return GaussianMixin.act(self, inputs, "policy")

    def compute(self, inputs, role):
        x = inputs["states"][:, :self.num_observations]
        return self.mean_layer(self.net(x)), self.log_std_parameter, {}
        
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, hidden_size=256): #observation_space init like state_space
        Model.__init__(self, observation_space, action_space, device) #observation_space init like state_space
        DeterministicMixin.__init__(self, clip_actions)
        print("Value: observation_space", observation_space)
        print("Value: action_space", action_space)
        print("Value: num_observations", self.num_observations)
        self.net = nn.Sequential(nn.Linear(self.num_observations, hidden_size), #num_observations init like num_states
                                 nn.ReLU(), #Also Tanh() or ELU()
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(), #Also Tanh() or ELU()
                                 nn.Linear(hidden_size, hidden_size // 2),
                                 nn.ReLU()  #Also Tanh() or ELU()
                                 )
        self.value_layer = nn.Linear(hidden_size // 2, 1)

    def act(self, inputs, role):
        return DeterministicMixin.act(self, inputs, "value")

    def compute(self, inputs, role):
        x = inputs["states"][:, :self.num_observations]
        return self.value_layer(self.net(x)), {}

