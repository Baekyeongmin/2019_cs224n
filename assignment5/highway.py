#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch.nn as nn


class Highway(nn.Module):

    def __init__(self, embed_size):
        super(Highway, self).__init__()
        self.projection = nn.Linear(in_features=embed_size,
                                    out_features=embed_size,
                                    bias=True)
        self.gate = nn.Linear(in_features=embed_size,
                              out_features=embed_size,
                              bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_proj = self.relu(self.projection(x))
        x_gate = self.sigmoid(self.gate(x))
        x_highway = x_gate * x_proj + (1 - x_gate) * x
        return x_highway


if __name__ == "__main__":
    import torch
    test_tensor = torch.rand(30,100,1024)
    highway_layer = Highway(1024)
    print(highway_layer.forward(test_tensor).size())


### END YOUR CODE 

