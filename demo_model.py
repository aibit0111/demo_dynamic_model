from distutils.log import debug
from fileinput import filename
from flask import *  
import pandas as pd
import torch
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.nn.functional import softplus

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.infer.autoguide import AutoDiagonalNormal, AutoDelta, AutoLowRankMultivariateNormal, AutoMultivariateNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from tqdm.auto import trange, tqdm
from pyro import poutine
import json
from flask import Flask, render_template, request, redirect, url_for, session
from torch.utils.data import TensorDataset, DataLoader
import dash
from dash import dcc
from dash import html
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly


class NeuralNetworkBlock(PyroModule):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NeuralNetworkBlock, self).__init__()

        self.mu = torch.Tensor([0.])
        self.std = torch.Tensor([0.5])
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        # Define the layers
        if len(hidden_sizes):
            self.linear_0 = PyroModule[nn.Linear](self.input_size, self.hidden_sizes[0])
            self.linear_0.weight = PyroSample(dist.Normal(self.mu,  self.std).expand([ self.hidden_sizes[0], self.input_size]).to_event(2))
            self.linear_0.bias = PyroSample(dist.Normal(self.mu,  self.std).expand([self.hidden_sizes[0]]).to_event(1))

            for i in range(1, len(hidden_sizes)):
                setattr(self, 'linear_{}'.format(i), PyroModule[nn.Linear](self.hidden_sizes[i-1], self.hidden_sizes[i]))
                getattr(self, 'linear_{}'.format(i)).weight = PyroSample(dist.Normal(self.mu,  self.std).expand([self.hidden_sizes[i], self.hidden_sizes[i-1]]).to_event(2))
                getattr(self, 'linear_{}'.format(i)).bias = PyroSample(dist.Normal(self.mu,  self.std).expand([self.hidden_sizes[i]]).to_event(1))

            self.linear_out = PyroModule[nn.Linear](self.hidden_sizes[-1], self.output_size)
            self.linear_out.weight = PyroSample(dist.Normal(self.mu, self.std).expand([self.output_size, self.hidden_sizes[-1]]).to_event(2))
            self.linear_out.bias = PyroSample(dist.Normal(self.mu,  self.std).expand([self.output_size]).to_event(1))

        else:
            self.linear_out = PyroModule[nn.Linear](self.input_size, self.output_size)
            self.linear_out.weight = PyroSample(dist.Normal(self.mu,  self.std).expand([self.output_size, self.input_size]).to_event(2))
            self.linear_out.bias = PyroSample(dist.Normal(self.mu,  self.std).expand([self.output_size]).to_event(1))

    def forward(self, x):
        logits = x
        for i in range(len(self.hidden_sizes)):
            logits = torch.relu(getattr(self, 'linear_{}'.format(i))(logits))
        logits = self.linear_out(logits)

        return logits

class ManualModelBlock(PyroModule):
    def __init__(self, default_mu=0.0, default_std=0.5, bias_mu=0.0, bias_std=0.5):
        super().__init__()
        self.mu = torch.Tensor([default_mu])
        self.std = torch.Tensor([default_std])
        self.bias_mu = torch.Tensor([bias_mu])
        self.bias_std = torch.Tensor([bias_std])

    def forward(self, variable, parameter_name, bias, categorical, function):
        if bias:
            bias = pyro.sample(parameter_name, dist.Normal(self.bias_mu,  self.bias_std))
            result = bias
            return result

        if categorical:
            parameter_name = pyro.sample(parameter_name, dist.Normal(self.mu,  self.std).expand([2]).to_event(1))
            result = parameter_name[variable]
            return result

        parameter_name = pyro.sample(parameter_name, dist.Normal(self.mu,  self.std).expand([1]).to_event(1))

        if function == 0:
            result = variable * parameter_name
        elif function == 1:
            result = variable ** 2 * parameter_name
        elif function == 2:
            result = variable ** 3 * parameter_name
        elif function == 3:
            result = torch.sqrt(variable) * parameter_name
        elif function == 4:
            result = torch.exp(variable) * parameter_name
        elif function == 5:
            result = torch.log(variable) * parameter_name
        else:
            result = torch.sin(variable) * parameter_name
        
        return result

indexing_varibale_parameter = 0

class Model(PyroModule):
    def __init__(self, equation, response_var_index, default_mu=0.0, default_std=0.5, bias_mu=0.0, bias_std=0.5):
        super().__init__()
        self.equation = equation
        self.linear_layer = ManualModelBlock(default_mu=default_mu, default_std=default_std, bias_mu=bias_mu, bias_std=bias_std)

        self.neural_network_variable = []
        self.input_all_neural_network = []
        self.hidden_all_neural_network = []
        self.output_all_neural_network = []
        self.manual_network = []
        self.bias = []

        self.response_var_index = response_var_index

        for i in range(len(self.equation)):

            if len(self.equation[i]) == 1:
                self.bias = self.equation[i]
                print(self.bias)
                continue
            
            block_type = self.equation[i][-1]
            if block_type == 0 or block_type == 2: # Means manual network
            #print(i)

                if block_type == 2:
                    self.categorical = 1
                else:
                    self.categorical = 0

                temp_manual_block = []
                data_node_index = self.equation[i][0]
                data_node_id = list(nodes_name_id)[data_node_index]
                parameter_node_id = list(nodes_name_id)[self.equation[i][1]]
                data_node_name = nodes_name_id[node_id]
                parameter_name = all_nodes[parameter_node_id]['text']
                if len(parameter_name) == 0:
                    parameter_name = "alpha" + str(i)
                temp_manual_block.append(data_node_index)
                temp_manual_block.append(parameter_name)
                temp_manual_block.append(self.categorical)
                self.manual_network.append(temp_manual_block)

            elif block_type == 1: # Means Neural network
                temp_nn_block = []
                data_node_index = self.equation[i][:-2] 
                parameter_name = self.equation[i][-2]
                if len(parameter_name) == 0:
                    parameter_name = "neural" + str(i)

                data_node_index.append(parameter_name)
                self.neural_network_variable.append(data_node_index)
                self.input_all_neural_network.append(len(self.equation[i])-2)
                self.hidden_all_neural_network.append([(len(self.equation[i])-2) * 2, (len(self.equation[i])-2) * 2])
                self.output_all_neural_network.append(1)

        for i in range(len(self.neural_network_variable)):
            input_size_index = self.input_all_neural_network[i]
            hidden_sizes_index = self.hidden_all_neural_network[i]
            output_size_index = self.output_all_neural_network[i]
            #print(i, input_size_index, hidden_sizes_index, output_size_index)
            # print("self.neural_network_variable[i]", self.neural_network_variable[i])
            setattr(self, "NeuralNetworkBlock_{}".format(self.neural_network_variable[i][-1]), NeuralNetworkBlock(input_size_index, hidden_sizes_index, output_size_index))

        print(self.manual_network)

    def forward(self, X):

        y = X[:, self.response_var_index]
        result = torch.zeros(X.shape[0])

        for i in range(len(self.neural_network_variable)):
            variable_index = self.neural_network_variable[i][:-1]
            parameter_name = self.neural_network_variable[i][-1]
            variable_value = X[:,variable_index]
            temp = getattr(self, "NeuralNetworkBlock_{}".format(parameter_name))(variable_value).squeeze()
            parameter_name = pyro.deterministic(parameter_name, temp)
            result = result + temp

        for i in range(len(self.manual_network)):
            categorical = self.manual_network[i][2]
            variable_index = self.manual_network[i][0]
            if categorical == 1:
                variable_value = X[:,variable_index].long()
            else:
                variable_value = X[:,variable_index]
            parameter_name = self.manual_network[i][1]
            result = result + self.linear_layer(variable_value, parameter_name, 0, categorical, 0)

        if len(self.bias) == 1:
            parameter_name = all_nodes[list(nodes_name_id)[self.bias[0]]]['text']
            # print("parameter_name", parameter_name)
            if len(parameter_name) == 0:
                parameter_name = "global_bias"
            # print("variable value", variable_value, variable_value.unique())
            result = result + self.linear_layer(None, parameter_name, 1, None, 0)

        #print(result.shape)

        #print(result.shape)
        p = pyro.deterministic('p', torch.sigmoid(result))

        with pyro.plate("N", X.shape[0]):
            obs = pyro.sample("status", dist.Bernoulli(probs=p), obs=y)
        return obs
