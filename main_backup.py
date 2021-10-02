#!/usr/bin/env python
# coding: utf-8
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import model
import anomaly_detection
from utils import parse_args
import data as od
from dataloader import SignalDataset

logging.basicConfig(filename='train.log', level=logging.DEBUG)


def critic_x_iteration(sample):
    optim_cx.zero_grad()

    x = sample['value'].view(1, batch_size, signal_shape)
    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)
    critic_score_valid_x = torch.mean(torch.ones(valid_x.shape) * valid_x) #Wasserstein Loss

    #The sampled z are the anomalous points - points deviating from actual distribution of z (obtained through encoding x)
    z = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
    x_ = decoder(z)
    fake_x = critic_x(x_)
    fake_x = torch.squeeze(fake_x)
    critic_score_fake_x = torch.mean(torch.ones(fake_x.shape) * fake_x)  #Wasserstein Loss

    alpha = torch.rand(x.shape)
    ix = Variable(alpha * x + (1 - alpha) * x_) #Random Weighted Average
    ix.requires_grad_(True)
    v_ix = critic_x(ix)
    v_ix.mean().backward()
    gradients = ix.grad
    #Gradient Penalty Loss
    gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))

    #Critic has to maximize Cx(Valid X) - Cx(Fake X).
    #Maximizing the above is same as minimizing the negative.
    wl = critic_score_fake_x - critic_score_valid_x
    loss = wl + gp_loss
    loss.backward()
    optim_cx.step()

    return loss

def critic_z_iteration(sample):
    optim_cz.zero_grad()

    x = sample['value'].view(1, batch_size, signal_shape)
    z = encoder(x)
    valid_z = critic_z(z)
    valid_z = torch.squeeze(valid_z)
    critic_score_valid_z = torch.mean(torch.ones(valid_z.shape) * valid_z)

    z_ = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
    fake_z = critic_z(z_)
    fake_z = torch.squeeze(fake_z)
    critic_score_fake_z = torch.mean(torch.ones(fake_z.shape) * fake_z) #Wasserstein Loss

    wl = critic_score_fake_z - critic_score_valid_z

    alpha = torch.rand(z.shape)
    iz = Variable(alpha * z + (1 - alpha) * z_) #Random Weighted Average
    iz.requires_grad_(True)
    v_iz = critic_z(iz)
    v_iz.mean().backward()
    gradients = iz.grad
    gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))

    loss = wl + gp_loss
    loss.backward()
    optim_cz.step()

    return loss

def encoder_iteration(sample):
    optim_enc.zero_grad()

    x = sample['value'].view(1, batch_size, signal_shape)
    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)
    critic_score_valid_x = torch.mean(torch.ones(valid_x.shape) * valid_x) #Wasserstein Loss

    z = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
    x_ = decoder(z)
    fake_x = critic_x(x_)
    fake_x = torch.squeeze(fake_x)
    critic_score_fake_x = torch.mean(torch.ones(fake_x.shape) * fake_x)

    enc_z = encoder(x)
    gen_x = decoder(enc_z)

    mse = mse_loss(x.float(), gen_x.float())
    loss_enc = mse + critic_score_valid_x - critic_score_fake_x
    loss_enc.backward(retain_graph=True)
    optim_enc.step()

    return loss_enc

def decoder_iteration(sample):
    optim_dec.zero_grad()

    x = sample['value'].view(1, batch_size, signal_shape)
    z = encoder(x)
    valid_z = critic_z(z)
    valid_z = torch.squeeze(valid_z)
    critic_score_valid_z = torch.mean(torch.ones(valid_z.shape) * valid_z)

    z_ = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
    fake_z = critic_z(z_)
    fake_z = torch.squeeze(fake_z)
    critic_score_fake_z = torch.mean(torch.ones(fake_z.shape) * fake_z)

    enc_z = encoder(x)
    gen_x = decoder(enc_z)

    mse = mse_loss(x.float(), gen_x.float())
    loss_dec = mse + critic_score_valid_z - critic_score_fake_z
    loss_dec.backward(retain_graph=True)
    optim_dec.step()

    return loss_dec


def train(n_epochs=2000):
    logging.debug('Starting training')
    cx_epoch_loss = list()
    cz_epoch_loss = list()
    encoder_epoch_loss = list()
    decoder_epoch_loss = list()

    for epoch in range(n_epochs):
        logging.debug('Epoch {}'.format(epoch))
        n_critics = 5

        cx_nc_loss = list()
        cz_nc_loss = list()

        for i in range(n_critics):
            cx_loss = list()
            cz_loss = list()

            for batch, sample in enumerate(train_loader):
                loss = critic_x_iteration(sample)
                cx_loss.append(loss)

                loss = critic_z_iteration(sample)
                cz_loss.append(loss)

            cx_nc_loss.append(torch.mean(torch.tensor(cx_loss)))
            cz_nc_loss.append(torch.mean(torch.tensor(cz_loss)))

        logging.debug('Critic training done in epoch {}'.format(epoch))
        encoder_loss = list()
        decoder_loss = list()

        for batch, sample in enumerate(train_loader):
            enc_loss = encoder_iteration(sample)
            dec_loss = decoder_iteration(sample)
            encoder_loss.append(enc_loss)
            decoder_loss.append(dec_loss)

        cx_epoch_loss.append(torch.mean(torch.tensor(cx_nc_loss)))
        cz_epoch_loss.append(torch.mean(torch.tensor(cz_nc_loss)))
        encoder_epoch_loss.append(torch.mean(torch.tensor(encoder_loss)))
        decoder_epoch_loss.append(torch.mean(torch.tensor(decoder_loss)))
        logging.debug('Encoder decoder training done in epoch {}'.format(epoch))
        logging.debug('critic x loss {:.3f} critic z loss {:.3f} \nencoder loss {:.3f} decoder loss {:.3f}\n'.format(cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))
        print('Encoder decoder training done in epoch {}'.format(epoch))
        print('critic x loss {:.3f} critic z loss {:.3f} \nencoder loss {:.3f} decoder loss {:.3f}\n'.format(cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))

        if epoch % 10 == 0:
            torch.save(encoder.state_dict(), encoder.encoder_path)
            torch.save(decoder.state_dict(), decoder.decoder_path)
            torch.save(critic_x.state_dict(), critic_x.critic_x_path)
            torch.save(critic_z.state_dict(), critic_z.critic_z_path)

if __name__ == "__main__":
    params = parse_args()

    train_data = od.load_signal('nyc_taxi')
    test_data = od.load_signal('nyc_taxi')

    train_dataset = SignalDataset(path='./data/{}-train.csv'.format(params.signal))
    test_dataset = SignalDataset(path='./data/{}-test.csv'.format(params.signal))
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    logging.info('Number of train datapoints is {}'.format(len(train_dataset)))
    logging.info('Number of samples in train dataset {}'.format(len(train_dataset)))

    lr = 0.0005

    signal_shape = 100
    latent_space_dim = 20
    encoder_path = 'models/encoder.pt'
    decoder_path = 'models/decoder.pt'
    critic_x_path = 'models/critic_x.pt'
    critic_z_path = 'models/critic_z.pt'
    
    encoder = model.Encoder(encoder_path, signal_shape)
    decoder = model.Decoder(decoder_path, signal_shape)
    critic_x = model.CriticX(critic_x_path, signal_shape)
    critic_z = model.CriticZ(critic_z_path)

    mse_loss = torch.nn.MSELoss()

    optim_enc = optim.Adam(encoder.parameters(), lr=lr, betas=(0.9, 0.999))
    optim_dec = optim.Adam(decoder.parameters(), lr=lr, betas=(0.9, 0.999))
    optim_cx = optim.Adam(critic_x.parameters(), lr=lr, betas=(0.9, 0.999))
    optim_cz = optim.Adam(critic_z.parameters(), lr=lr, betas=(0.9, 0.999))

    train(n_epochs=params.epochs)

    anomaly_detection.test(test_loader, encoder, decoder, critic_x)
