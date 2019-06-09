#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn as nn
import torch


class CNN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(CNN, self).__init__()

        self.conv_1d = nn.Conv1d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, input):
        x_conv_out = []

        for i in range(input.size()[0]):
            x_conv = self.conv_1d(input[i, :, :, :])
            x_conv = self.relu(x_conv)
            x_conv = self.max_pool(x_conv)
            x_conv = torch.squeeze(x_conv, 2)
            x_conv_out.append(x_conv)

        return torch.stack(x_conv_out, dim=0)


### END YOUR CODE

