#!/usr/bin/env python
# coding: utf-8

import os
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from hyperspace.hyrnn_nets import MobiusLinear, MobiusDist2Hyperplane
import geoopt.manifolds.stereographic.math as gmath

class Encoder(nn.Module):

    def __init__(self, encoder_path='', signal_shape=100, latent_space_dim=20, hyperbolic=False, sequence_shape=100):
        super(Encoder, self).__init__()
        self.signal_shape = signal_shape
        self.sequence_shape = sequence_shape
        self.latent_space_dim = latent_space_dim
        self.lstm = nn.LSTM(input_size=self.signal_shape, hidden_size=50, num_layers=1, bidirectional=True)
        self.dense = nn.Linear(in_features=100, out_features=self.latent_space_dim)
        self.encoder_path = encoder_path
        self.hyperbolic = False
        if self.hyperbolic:
          self.hyperbolic_linear = MobiusLinear(self.latent_space_dim,
                                          self.latent_space_dim,
                                          # This computes an exmap0 after the operation, where the linear
                                          # operation operates in the Euclidean space.
                                          hyperbolic_input=False,
                                          hyperbolic_bias=True,
                                          nonlin=None,  # For now
                                          fp64_hyper=False
                                          )

    def forward(self, x):
        x = x.view(self.sequence_shape, -1, self.signal_shape).float()
        x, (hn, cn) = self.lstm(x)
        x = self.dense(x)
        if self.hyperbolic:
          hyper_gen = self.hyperbolic_linear(x.view(-1, 20))
          eucl = gmath.logmap0(hyper_gen, k=torch.tensor(-1.), dim=1).float()

          return (eucl.view(1,-1,20),hyper_gen)

        return (x)

class Decoder(nn.Module):
    def __init__(self, decoder_path, signal_shape=100, latent_space_dim=20, hyperbolic=False):
        super(Decoder, self).__init__()
        self.signal_shape = signal_shape
        self.latent_space_dim = latent_space_dim
        self.dense1 = nn.Linear(in_features=self.latent_space_dim, out_features=50) 
        self.lstm = nn.LSTM(input_size=50, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=True)
        self.dense2 = nn.Linear(in_features=128, out_features=self.signal_shape)
        self.decoder_path = decoder_path
        self.hyperbolic = hyperbolic
        if self.hyperbolic:
          self.hyperbolic_linear = MobiusLinear(self.signal_shape,
                                          self.signal_shape,
                                          # This computes an exmap0 after the operation, where the linear
                                          # operation operates in the Euclidean space.
                                          hyperbolic_input=False,
                                          hyperbolic_bias=True,
                                          nonlin=None,  # For now
                                          fp64_hyper=False
                                          )
          # self.dist = MobiusDist2Hyperplane(100,100,fp64_hyper=False)
          # self.lin1 = nn.Linear(100, 32)
          # self.relu = nn.ReLU()
          # self.lin2 = nn.Linear(32, 100)

    def forward(self, x):
        x = self.dense1(x)
        x, (hn, cn) = self.lstm(x)
        x = self.dense2(x)
        if self.hyperbolic:
          hyper_gen_init = self.hyperbolic_linear(x.view(-1, self.signal_shape))
          # hyper_gen_dist = self.dist(hyper_gen_init)
          # eucl = gmath.logmap0(hyper_gen_init, k=torch.tensor(-1.), dim=1).float()
          # # eucl = self.lin1(eucl)
          # eucl = self.lin2(self.relu(self.lin1(eucl)))
          return (hyper_gen_init.view(1,-1,self.signal_shape),x)
          
          # return (eucl.view(1,-1,self.signal_shape),hyper_gen)

          # return (hyper_gen.view(1,-1,self.signal_shape),hyper_gen)

        return (x)

class CriticX(nn.Module):
    def __init__(self, critic_x_path, signal_shape=10, latent_space_dim=20, sequence_shape=100):
        super(CriticX, self).__init__()
        self.signal_shape = signal_shape
        self.sequence_shape = sequence_shape
        self.latent_space_dim  =  latent_space_dim
        self.dense1 = nn.Linear(in_features=self.signal_shape, out_features=self.latent_space_dim)
        self.dense2 = nn.Linear(in_features=self.latent_space_dim, out_features=1)
        self.critic_x_path = critic_x_path

    def forward(self, x):
        x = x.view(1, -1, self.signal_shape).float()
        x = self.dense1(x)
        x = self.dense2(x)
        return (x)

class CriticZ(nn.Module):
    def __init__(self, critic_z_path,latent_space_dim=20):
        super(CriticZ, self).__init__()
        self.latent_space_dim = latent_space_dim
        self.dense1 = nn.Linear(in_features=self.latent_space_dim, out_features=1)
        self.critic_z_path = critic_z_path

    def forward(self, x):
        x = self.dense1(x)
        return (x)

def unroll_signal(self, x):
    x = np.array(x).reshape(100)
    return np.median(x)

def test(self):
    """
    Returns a dataframe with original value, reconstructed value, reconstruction error, critic score
    """
    df = self.test_dataset.copy()
    X_ = list()

    RE = list()  #Reconstruction error
    CS = list()  #Critic score

    for i in range(0, df.shape[0]):
        x = df.rolled_signal[i]
        x = tf.reshape(x, (1, 100, 1))
        z = encoder(x)
        z = tf.expand_dims(z, axis=2)
        x_ = decoder(z)

        re = dtw_reconstruction_error(tf.squeeze(x_).numpy(), tf.squeeze(x).numpy()) #reconstruction error
        cs = critic_x(x)
        cs = tf.squeeze(cs).numpy()
        RE.append(re)
        CS.append(cs)

        x_ = unroll_signal(x_)

        X_.append(x_)

    df['generated_signals'] = X_

    return df
