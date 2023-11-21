#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch.nn as nn

from hyperspace.hyrnn_nets import MobiusLinear


class Encoder(nn.Module):
    def __init__(self, signal_shape=100, latent_space_dim=20, hyperbolic=False):
        super(Encoder, self).__init__()
        self.signal_shape = signal_shape
        self.latent_space_dim = latent_space_dim
        self.lstm = nn.LSTM(
            input_size=self.signal_shape,
            hidden_size=50,
            num_layers=1,
            bidirectional=True,
        )
        self.dense = nn.Linear(in_features=100, out_features=self.latent_space_dim)

    def forward(self, x):
        x = x.view(1, -1, self.signal_shape).float()
        x, (hn, cn) = self.lstm(x)
        x = self.dense(x)
        return x


class Decoder(nn.Module):
    def __init__(self, signal_shape=100, latent_space_dim=20, hyperbolic=False):
        super(Decoder, self).__init__()
        self.signal_shape = signal_shape
        self.latent_space_dim = latent_space_dim
        self.dense1 = nn.Linear(in_features=self.latent_space_dim, out_features=50)
        self.lstm = nn.LSTM(
            input_size=50, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=True
        )
        self.dense2 = nn.Linear(in_features=128, out_features=self.signal_shape)
        self.tanh = nn.Tanh()
        self.hyperbolic = hyperbolic
        if self.hyperbolic:
            self.hyperbolic_linear = MobiusLinear(
                self.signal_shape,
                self.signal_shape,
                # This computes an exmap0 after the operation, where the linear
                # operation operates in the Euclidean space.
                hyperbolic_input=False,
                hyperbolic_bias=True,
                nonlin=None,  # For now
                fp64_hyper=False,
            )
            # self.dist = MobiusDist2Hyperplane(100,100,fp64_hyper=False)
            # self.lin1 = nn.Linear(100, 32)
            # self.relu = nn.ReLU()
            # self.lin2 = nn.Linear(32, 100)

    def forward(self, x):
        x = self.dense1(x)
        x, (hn, cn) = self.lstm(x)
        x = self.dense2(x)
        x = self.tanh(x)
        if self.hyperbolic:
            hyper_gen_init = self.hyperbolic_linear(x.view(-1, self.signal_shape))
            return (hyper_gen_init.view(1, -1, self.signal_shape), x)

        return x


class CriticX(nn.Module):
    def __init__(self, signal_shape=10, latent_space_dim=20):
        super(CriticX, self).__init__()
        self.signal_shape = signal_shape
        self.latent_space_dim = latent_space_dim
        self.dropout = nn.Dropout(p=0.25)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dense1 = nn.Linear(
            in_features=self.signal_shape, out_features=self.latent_space_dim
        )
        self.dense2 = nn.Linear(
            in_features=self.latent_space_dim, out_features=self.latent_space_dim
        )
        self.dense3 = nn.Linear(
            in_features=self.latent_space_dim, out_features=self.latent_space_dim
        )
        self.dense4 = nn.Linear(
            in_features=self.latent_space_dim, out_features=self.latent_space_dim
        )
        self.dense5 = nn.Linear(in_features=self.latent_space_dim, out_features=1)

    def forward(self, x):
        x = x.view(1, -1, self.signal_shape).float()
        x = self.dense1(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.dense3(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.dense4(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.dense5(x)
        return x


class CriticZ(nn.Module):
    def __init__(self, latent_space_dim=20):
        super(CriticZ, self).__init__()
        self.latent_space_dim = latent_space_dim
        self.dense1 = nn.Linear(
            in_features=self.latent_space_dim, out_features=self.latent_space_dim
        )
        self.dense2 = nn.Linear(
            in_features=self.latent_space_dim, out_features=self.latent_space_dim
        )
        self.dense3 = nn.Linear(in_features=self.latent_space_dim, out_features=1)
        self.dropout = nn.Dropout(p=0.2)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.dense1(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.dense3(x)

        return x


def unroll_signal(self, x):
    x = np.array(x).reshape(100)
    return np.median(x)
