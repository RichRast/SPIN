import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pdb


# MLP feature extractor
class MLP(nn.Module):
    def __init__(
            self, input_size, hidden_layer_sizes, output_size,
            dropout_prob=None):
        super(MLP, self).__init__()
        fc_layers = []
        all_layer_sizes = [input_size] + hidden_layer_sizes
        for layer_size_idx in range(len(all_layer_sizes) - 1):
            fc_layers.append(
                nn.Linear(all_layer_sizes[layer_size_idx],
                          all_layer_sizes[layer_size_idx + 1]))

        self.fc_layers = nn.ModuleList(fc_layers)
        self.output_layer = nn.Linear(
            hidden_layer_sizes[-1], output_size)

        if dropout_prob is not None:
            self.dropout = torch.nn.Dropout(p=dropout_prob)
        else:
            self.dropout = None

    def forward(self, x):
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = F.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        output = self.output_layer(x)
        return output

class MLPClassificationModel(nn.Module):
    def __init__(self, feature_extractor, batch_size, num_dim):
        super(MLPClassificationModel, self).__init__()
        pass

    def forward(self, x):
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = F.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        output = self.output_layer(x)
        return output

class MLPRegressionModel(nn.Module):
    def __init__(self, feature_extractor, input_size, hidden_layer_sizes, output_size, dropout_prob):
        super(MLPRegressionModel, self).__init__()
        self.feature_extractor=feature_extractor
        fc_layers = []
        all_layer_sizes = [input_size] + hidden_layer_sizes
        for layer_size_idx in range(len(all_layer_sizes) - 1):
            fc_layers.append(
                nn.Linear(all_layer_sizes[layer_size_idx],
                          all_layer_sizes[layer_size_idx + 1]))

        self.fc_layers = nn.ModuleList(fc_layers)
        self.output_layer = nn.Linear(
            hidden_layer_sizes[-1], output_size)

        if dropout_prob is not None:
            self.dropout = torch.nn.Dropout(p=dropout_prob)
        else:
            self.dropout = None

    def forward(self, x):
        features=self.feature_extractor(x)
        for fc_layer in self.fc_layers:
            features = fc_layer(features)
            features = F.relu(features)

        if self.dropout is not None:
            features = self.dropout(features)

        output = self.output_layer(features)
        return output
