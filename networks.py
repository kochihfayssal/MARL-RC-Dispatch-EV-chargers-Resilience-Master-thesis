import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class PPOCritic(nn.Module):
    def __init__(self, critic_input_shape, args, layer_norm=True):
        super(PPOCritic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(critic_input_shape, self.args.rnn_hidden_dim)
        # self.fc2 = nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.args.rnn_hidden_dim, 1)

        if layer_norm:
            self.layer_init(self.fc1)
            # self.layer_init(self.fc2)
            self.layer_init(self.fc2, std=1.0)
    
    @staticmethod
    def layer_init(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, inputs, hidden):
        x = F.relu(self.fc1(inputs))
        # y = F.relu(self.fc2(x))
        h_in = hidden.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        v = self.fc2(h)
        return v, h

class PPOActor(nn.Module):
    def __init__(self, input_shape, args, layer_norm=True):
        super(PPOActor, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, self.args.rnn_hidden_dim)
        # self.fc2 = nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.args.rnn_hidden_dim, self.args.n_actions)
    
        if layer_norm:
            self.layer_init(self.fc1, std=1.0)
            self.layer_init(self.fc2, std=1.0)
            # self.layer_init(self.fc3, std=1.0)
        
    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        # y = F.relu(self.fc2(x))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

#################################################################################

class TRPOCritic(nn.Module):
    def __init__(self, critic_input_shape, args, layer_norm=True):
        super(TRPOCritic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(critic_input_shape, self.args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.args.rnn_hidden_dim, 1)

        if layer_norm:
            self.layer_init(self.fc1)
            self.layer_init(self.fc2, std=1.0)
    
    @staticmethod
    def layer_init(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, inputs, hidden): #hidden
        x = F.relu(self.fc1(inputs))
        h_in = hidden.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in) 
        v = self.fc2(h)
        return v , h
    
class TRPOActor(nn.Module):
    def __init__(self, input_shape, args, layer_norm=True):
        super(TRPOActor, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, self.args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.args.rnn_hidden_dim, self.args.n_actions)
    
        if layer_norm:
                self.layer_init(self.fc1, std=1.0)
                self.layer_init(self.fc2, std=1.0)
        
    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    
    def forward(self, obs, hidden_state): #hidden_state
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q , h

#################################################################################

class A2CCritic(nn.Module):
    def __init__(self, critic_input_shape, args, layer_norm=True):
        super(A2CCritic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(critic_input_shape, self.args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.args.rnn_hidden_dim, 1)

        if layer_norm:
            self.layer_init(self.fc1)
            self.layer_init(self.fc2, std=1.0)
    
    @staticmethod
    def layer_init(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, inputs, hidden):
        x = F.relu(self.fc1(inputs))
        h_in = hidden.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        v = self.fc2(h)
        return v, h

class A2CActor(nn.Module):
    def __init__(self, input_shape, args, layer_norm=True):
        super(A2CActor, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, self.args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.args.rnn_hidden_dim, self.args.n_actions)
    
        if layer_norm:
            self.layer_init(self.fc1, std=1.0)
            self.layer_init(self.fc2, std=1.0)
        
    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

#################################################################################

class DDPGCritic(nn.Module):
    def __init__(self, critic_input_shape, args, layer_norm=True):
        super(DDPGCritic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(critic_input_shape, self.args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.args.rnn_hidden_dim, self.args.n_actions)

        if layer_norm:
            self.layer_init(self.fc1)
            self.layer_init(self.fc2, std=1.0)
    
    @staticmethod
    def layer_init(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, inputs, hidden):
        x = F.relu(self.fc1(inputs))
        h_in = hidden.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        v = self.fc2(h)
        return v, h
    
class DDPGActor(nn.Module):
    def __init__(self, input_shape, args, layer_norm=True):
        super(DDPGActor, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, self.args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.args.rnn_hidden_dim, self.args.n_actions)
    
        if layer_norm:
            self.layer_init(self.fc1, std=1.0)
            self.layer_init(self.fc2, std=1.0)
        
    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h