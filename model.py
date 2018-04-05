import torch
import torch.nn as nn
from collections import OrderedDict
'''
MLP
multi-layer-perception

for mnist data is 28*28
so the input dim is 784

the mnist data is the picture of 0-9, the class num is 10

n_hiddens if for dim of each hidden layer

nn.Sequential
    the sequential container
    input is an ordered dict of modules or list of modules
'''

class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class, type_act, dropout=0.2):
        '''
        :param input_dims: int, 784 as default
        :param n_hiddens: list of int, dim for each hidden layers
        :param n_class: int, 10 as default
        :param type_act: str, like 'relu', 'sigmoid'
        '''
        super(MLP, self).__init__()
        self.input_dims = input_dims
        self.n_hiddens = n_hiddens
        self.n_class = n_class
        self.dropout = dropout
        tmp_dim = input_dims
        multi_layers = OrderedDict()

        # for activation function
        dic_act =      {'relu':nn.ReLU(), 'sig':nn.Sigmoid(), 'tanh':nn.Tanh()}

        for idx, n_hidden in enumerate(n_hiddens):
            multi_layers['fc_{}'.format(idx+1)] = nn.Linear(tmp_dim, n_hidden)
            multi_layers['{}_{}'.format(type_act, idx+1)] = dic_act[type_act]
            multi_layers['drop_{}'.format(idx+1)] = nn.Dropout(dropout)
            tmp_dim = n_hidden
        multi_layers['output'] = nn.Linear(tmp_dim, n_class)

        self.mlp = nn.Sequential(multi_layers)

    def forward(self, input):
        '''
        :param input: shodld be [batchsize, 784]
        '''
        input = input.view(input.size(0), -1)
        # assert if dim is 784 (28*28)
        assert input.size(1) == self.input_dims
        return self.mlp.forward(input=input)
