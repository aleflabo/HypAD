#!usr/bin/bash python
# coding: utf-8
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import model_LSTM
import anomaly_detection_lstm
import anomaly_refactor

from utils import parse_args
import data as od
from dataloader import SignalDataset
from dataloader_multivariate import CasasDataset

from hyperspace.poincare_distance import poincare_distance
import geoopt
import geoopt.manifolds.stereographic.math as gmath
from hyperspace.losses import compute_mask,compute_scores
from hyperspace.utils import *

import torch.optim as optim

logging.basicConfig(filename='train.log', level=logging.DEBUG)

def critic_x_iteration(sample):
    optim_cx.zero_grad()
    x = sample.view(sequence_shape, batch_size, signal_shape)
    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)
    critic_score_valid_x = torch.mean(torch.ones(valid_x.shape).cuda() * valid_x) #Wasserstein Loss

    #The sampled z are the anomalous points - points deviating from actual distribution of z (obtained through encoding x)
    z = torch.Tensor(np.random.normal(size=(1,batch_size, latent_space_dim))).cuda()
    if decoder.hyperbolic:
      #hyperspace not used with the critics at the moment
      x_,eucl = decoder(z.cuda())
      # x_ = gmath.logmap0(x_, k=torch.tensor(-1.), dim=1).float()
    else: x_ = decoder(z.cuda())
    fake_x = critic_x(x_)
    fake_x = torch.squeeze(fake_x)
    critic_score_fake_x = torch.mean(torch.ones(fake_x.shape).cuda() * fake_x)  #Wasserstein Loss

    alpha = torch.rand(x.shape).cuda()
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
    loss.backward(retain_graph=True)
    optim_cx.step()

    return loss

    
def critic_z_iteration(sample):
    optim_cz.zero_grad()

    x = sample.view(sequence_shape, batch_size, signal_shape)
    if encoder.hyperbolic:
      z,_ = encoder(x)
    else: z = encoder(x)
    valid_z = critic_z(z.cuda())
    valid_z = torch.squeeze(valid_z)
    critic_score_valid_z = torch.mean(torch.ones(valid_z.shape).cuda() * valid_z)

    z_ = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1).cuda()
    fake_z = critic_z(z_)
    fake_z = torch.squeeze(fake_z)
    critic_score_fake_z = torch.mean(torch.ones(fake_z.shape).cuda() * fake_z) #Wasserstein Loss

    wl = critic_score_fake_z - critic_score_valid_z

    alpha = torch.rand(z.shape).cuda()
    iz = Variable(alpha * z + (1 - alpha) * z_) #Random Weighted Average
    iz.requires_grad_(True)
    v_iz = critic_z(iz)
    v_iz.mean().backward()
    gradients = iz.grad
    gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))

    loss = wl + gp_loss
    loss.backward(retain_graph=True)
    optim_cz.step()

    return loss

def encoder_iteration(sample):
    optim_enc.zero_grad()

    x = sample.view(sequence_shape, batch_size, signal_shape)
    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)
    critic_score_valid_x = torch.mean(torch.ones(valid_x.shape).cuda() * valid_x) #Wasserstein Loss

    z = torch.Tensor(np.random.normal(size=(1,batch_size, latent_space_dim))).cuda()
    # z = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1).cuda()
    if decoder.hyperbolic:
      x_, eucl = decoder(z)
      # x_ = gmath.logmap0(x_, k=torch.tensor(-1.), dim=1).float()
    else: x_ = decoder(z)
    fake_x = critic_x(x_)
    fake_x = torch.squeeze(fake_x)
    critic_score_fake_x = torch.mean(torch.ones(fake_x.shape).cuda() * fake_x)

    if encoder.hyperbolic:
      enc_z, hyper_z = encoder(x)

    else: enc_z = encoder(x)

    if decoder.hyperbolic:
      gen_x,eucl = decoder(enc_z)
      hyper_x = decoder.hyperbolic_linear(x.view(-1,signal_shape))

      #poincarè distance
      sqdist = torch.sum((gen_x - hyper_x) ** 2, dim=-1)
      squnorm = torch.sum(gen_x ** 2, dim=-1)
      sqvnorm = torch.sum(hyper_x ** 2, dim=-1)
      x_temp = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 1e-7
      dist = torch.acosh(x_temp)

      hyper_loss = torch.div(torch.sum(dist),batch_size)
      # mse=torch.Tensor([0])
      # loss_enc = hyper_loss + critic_score_valid_x - critic_score_fake_x

      mse = err_loss(eucl.float(), x.float())
      loss_enc = mse + hyper_loss + critic_score_valid_x - critic_score_fake_x

    else: 
      gen_x = decoder(enc_z)
      mse = err_loss(x.float(), gen_x.float())
      loss_enc = mse + critic_score_valid_x - critic_score_fake_x

    loss_enc.backward(retain_graph=True)
    optim_enc.step()

    return loss_enc


def decoder_iteration(sample):
    optim_dec.zero_grad()

    x = sample.view(sequence_shape, batch_size, signal_shape)
    z = torch.Tensor(np.random.normal(size=(1,batch_size, latent_space_dim))).cuda()
    if encoder.hyperbolic:
      #not used
      z_, _ = encoder(x)
      # mse_loss = err_loss(x.float(), gen_x.float())
    else: z_ = encoder(x)
    valid_z = critic_z(z)
    valid_z = torch.squeeze(valid_z)
    critic_score_valid_z = torch.mean(torch.ones(valid_z.shape).cuda() * valid_z)

    z_ = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1).cuda()
    fake_z = critic_z(z_)
    fake_z = torch.squeeze(fake_z)
    critic_score_fake_z = torch.mean(torch.ones(fake_z.shape).cuda() * fake_z)

    if encoder.hyperbolic:
        #not used at the moment
        enc_z, _ = encoder(x)
        # mse_loss = err_loss(x.float(), gen_x.float())
    else: enc_z = encoder(x)

    if decoder.hyperbolic:
      gen_x,eucl = decoder(enc_z)
      hyper_x = decoder.hyperbolic_linear(x.view(-1,signal_shape))

      sqdist = torch.sum((gen_x - hyper_x) ** 2, dim=-1)
      squnorm = torch.sum(gen_x ** 2, dim=-1)
      sqvnorm = torch.sum(hyper_x ** 2, dim=-1)
      x_temp = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 1e-7
      dist = torch.acosh(x_temp)

      hyper_loss = torch.div(torch.sum(dist),batch_size)
      # mse=torch.Tensor([0])
      # loss_dec = hyper_loss + critic_score_valid_z + critic_score_fake_z 

      mse = err_loss(eucl.float(), x.float())
      loss_dec = mse + hyper_loss + critic_score_valid_z - critic_score_fake_z

      loss_dec.backward(retain_graph=True)
      optim_dec.step()

      return loss_dec,hyper_loss,mse

      
    else: 
      gen_x = decoder(enc_z)
      mse_loss = err_loss(x.float(), gen_x.float())
      loss_dec = mse_loss + critic_score_valid_z - critic_score_fake_z

      loss_dec.backward()
      optim_dec.step()

      return loss_dec,0,mse_loss


def train_tadgan(encoder, decoder, critic_x, critic_z, n_epochs=2000, params=None):
    logging.debug('Starting training')
    cx_epoch_loss = list()
    cz_epoch_loss = list()
    encoder_epoch_loss = list()
    decoder_epoch_loss = list()
    hyp_dec_loss = list()
    eucl_dec_loss = list()
    err_loss = torch.nn.MSELoss()

    for epoch in range(n_epochs):

        logging.debug('Epoch {}'.format(epoch))
        n_critics = 5

        cx_nc_loss = list()
        cz_nc_loss = list()
        
        for param in decoder.parameters():
            param.requires_grad = False
        for param in encoder.parameters():
            param.requires_grad = False
        for param in critic_x.parameters():
            param.requires_grad = True
        for param in critic_z.parameters():
            param.requires_grad = True
        
        for i in range(n_critics):
            cx_loss = list()
            cz_loss = list()

            for batch, sample in enumerate(train_loader):
                loss = critic_x_iteration(sample.cuda())
                cx_loss.append(loss)

                loss = critic_z_iteration(sample.cuda())
                cz_loss.append(loss)

            cx_nc_loss.append(torch.mean(torch.tensor(cx_loss)))
            cz_nc_loss.append(torch.mean(torch.tensor(cz_loss)))


        for param in decoder.parameters():
            param.requires_grad = True
        for param in encoder.parameters():
            param.requires_grad = True
        for param in critic_x.parameters():
            param.requires_grad = False
        for param in critic_z.parameters():
            param.requires_grad = False

        logging.debug('Critic training done in epoch {}'.format(epoch))
        encoder_loss = list()
        decoder_loss = list()
        hyp_loss = list()
        mse_losss = list()
        for batch, sample in enumerate(train_loader):
            enc_loss = encoder_iteration(sample.cuda())
            dec_loss,hyper_loss,mse_loss = decoder_iteration(sample.cuda())
            encoder_loss.append(enc_loss)
            decoder_loss.append(dec_loss)
            if params.hyperbolic:
              hyp_loss.append(hyper_loss.float())
            mse_losss.append(mse_loss.float())

        cx_epoch_loss.append(torch.mean(torch.tensor(cx_nc_loss)))
        cz_epoch_loss.append(torch.mean(torch.tensor(cz_nc_loss)))
        encoder_epoch_loss.append(torch.mean(torch.tensor(encoder_loss)))
        decoder_epoch_loss.append(torch.mean(torch.tensor(decoder_loss)))
        if params.verbose:
          if params.hyperbolic:
            hyp_dec_loss.append(torch.mean(torch.tensor(hyp_loss)))
          eucl_dec_loss.append(torch.mean(torch.tensor(mse_losss)))
          # logging.debug('Encoder decoder training done in epoch {}'.format(epoch))
          # logging.debug('critic x loss {:.3f} critic z loss {:.3f} \nencoder loss {:.3f} decoder loss {:.3f}\n'.format(cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))
          
          print('Encoder decoder training done in epoch {}'.format(epoch))
          if params.hyperbolic: print('Hyperbolic loss {}'.format(hyp_dec_loss[-1]))
          print('Eucl mse loss {}'.format(eucl_dec_loss[-1]))
          print('critic x loss {:.3f} critic z loss {:.3f} \nencoder loss {:.3f} decoder loss {:.3f}\n'.format(cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))
          # print('critic x loss {:.3f} critic z loss {:.3f} \n decoder loss {:.3f}\n'.format(cx_epoch_loss[-1], cz_epoch_loss[-1], decoder_epoch_loss[-1]))

        # if (epoch % 10 == 0) or (epoch == (n_epochs-1)):
        #     torch.save(encoder.state_dict(), encoder.encoder_path)
        #     torch.save(decoder.state_dict(), decoder.decoder_path)
        #     torch.save(critic_x.state_dict(), critic_x.critic_x_path)
        #     torch.save(critic_z.state_dict(), critic_z.critic_z_path)


def train_lstm(model,optimizer,train_loader,test_loader,n_epochs=2000):
    logging.debug('Starting training')
    

    mse_loss = torch.nn.MSELoss()
    
    
    for epoch in range(n_epochs):
        logging.debug('Epoch {}'.format(epoch))
        train_losses = []
        model = model.train()
        for batch, sample in enumerate(train_loader):
            optimizer.zero_grad()
            sample = sample.cuda().reshape(-1,100)
            sample_pred = model(sample.float()).reshape(-1,100)
            loss = mse_loss(sample_pred.float(), sample.float())

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            
        val_losses = []
        model = model.eval()
        with torch.no_grad():
          for batch,(sample,_,_,_,_) in enumerate(test_loader):
            sample = sample.cuda().reshape(-1,100)
            sample_pred = model(sample.float()).reshape(-1,100)

            loss = mse_loss(sample_pred.float(), sample.float())
            val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        print('Epoch {} train_loss {} val_loss {}'.format(epoch, train_loss, val_loss))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), '/content/drive/MyDrive/TadGAN/models/trial.pt')

if __name__ == "__main__":
    params = parse_args()
    print('signal: {}'.format(params.signal))

    # dataset selection
    demo=False # demo is the taxy dataset present in the tadgan example. it has same data for train and test

    if params.dataset == 'CASAS':
        #this is a dataset used with the Team, not used at the moment
        path = '/content/drive/MyDrive/ASSISTED LIVING (e-linus)/Data&Experiments/DianeCookCASAS/'
        train_dataset = CasasDataset(seq_path=path+'sequences_2week_{}.pt'.format(params.signal), 
                                     gt_path=path+'ground_truth_2week_{}.pt'.format(params.signal), split=params.split)
        test_dataset = CasasDataset(seq_path=path+'sequences_2week_{}.pt'.format(params.signal), 
                                     gt_path=path+'ground_truth_2week_{}.pt'.format(params.signal), test=True)
    elif params.dataset == 'new_CASAS':
        path = './data/CASAS/new_dataset/'
        train_dataset = CasasDataset(seq_path=path+params.signal, 
                                     gt_path=path+params.signal, split=params.split, dataset=params.dataset)
        test_dataset = CasasDataset(seq_path=path+params.signal, 
                                     gt_path=path+params.signal, test=True, dataset=params.dataset)
        read_path = ''
    else:
        if params.unique_dataset:
            train_data = od.load_signal(params.signal)
            test_data = od.load_signal(params.signal)

            train_dataset = SignalDataset(path='./data/{}.csv'.format(params.signal),interval=params.interval)
            test_dataset = SignalDataset(path='./data/{}.csv'.format(params.signal),test=True,interval=params.interval)
            path = './data/{}.csv'
            read_path=path.format(params.signal)

        elif params.dataset in ['A1','A2','A3','A4']:

            train_dataset = SignalDataset(path='./data/YAHOO/{}Benchmark/{}.csv'.format(params.dataset,params.signal),interval=1)
            test_dataset = SignalDataset(path='./data/YAHOO/{}Benchmark/{}.csv'.format(params.dataset,params.signal),test=True,interval=1)
            read_path = './data/YAHOO/{}Benchmark/{}.csv'.format(params.dataset,params.signal)
            

        else:
            train_data = od.load_signal(params.signal +'-train')
            test_data = od.load_signal(params.signal +'-test')

            train_dataset = SignalDataset(path='./data/{}-train.csv'.format(params.signal),interval=params.interval)
            test_dataset = SignalDataset(path='./data/{}-test.csv'.format(params.signal),interval=params.interval,test=True)
            read_path='./data/{}-test.csv'.format(params.signal)

    batch_size = params.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True,shuffle=True,num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle=False,num_workers=6)

    logging.info('Number of train datapoints is {}'.format(len(train_dataset)))
    logging.info('Number of samples in train dataset {}'.format(len(train_dataset)))

    lr = params.lr
    lr_hyper = params.lr

    signal_shape = 1
    latent_space_dim = 20

    if params.model == 'lstm':
      #LSTM not used at the moment

      model = model_LSTM.RecurrentAutoencoder(batch_size=64, seq_len=100,n_features=signal_shape, embedding_dim=latent_space_dim)
      model = model.cuda()

      optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
      train_lstm(model, optimizer, train_loader, test_loader, n_epochs=params.epochs)
      
      known_anomalies = od.load_anomalies(params.signal)

      if demo:
        anomaly_detection_lstm.test_lstm(test_loader, model, known_anomalies, path='./data/{}.csv'.format(params.signal), signal = params.signal)
      else:
        anomaly_detection_lstm.test_lstm(test_loader, model, known_anomalies, path='./data/{}-test.csv'.format(params.signal),signal = params.signal)

    elif params.model == 'tadgan':
      encoder_path = 'models/encoder.pt'
      decoder_path = 'models/decoder.pt'
      critic_x_path = 'models/critic_x.pt'
      critic_z_path = 'models/critic_z.pt'

      signal_shape = params.signal_shape
      sequence_shape = 1

      encoder = model.Encoder(encoder_path, signal_shape, latent_space_dim, params.hyperbolic, sequence_shape=sequence_shape).cuda().train()
      decoder = model.Decoder(decoder_path, signal_shape, latent_space_dim, params.hyperbolic).cuda().train()
      critic_x = model.CriticX(critic_x_path, signal_shape, latent_space_dim,sequence_shape=sequence_shape).cuda().train()
      critic_z = model.CriticZ(critic_z_path, latent_space_dim).cuda().train()

      optim_enc = optim.Adam(encoder.parameters(), lr=lr, betas=(0.9, 0.999))
      optim_cx = optim.Adam(critic_x.parameters(), lr=lr, betas=(0.9, 0.999))

      if params.hyperbolic:
        optim_enc = geoopt.optim.RiemannianAdam(encoder.parameters(), lr=lr_hyper, weight_decay=1e-5, stabilize=10)
        optim_dec = geoopt.optim.RiemannianAdam(decoder.parameters(), lr=lr_hyper, weight_decay=1e-5, stabilize=10)
        # scheduler = optim.lr_scheduler.StepLR(optim_dec, step_size=7, gamma=0.1)

        # optim_cx = geoopt.optim.RiemannianAdam(critic_x.parameters(), lr=lr, weight_decay=1e-5, stabilize=10)
        # optim_cz = geoopt.optim.RiemannianAdam(critic_z.parameters(), lr=lr, weight_decay=1e-5, stabilize=10)
      else: 
        # model_params = list(encoder.parameters()) + list(decoder.parameters())
        optim_enc = optim.Adam(encoder.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_dec = optim.Adam(decoder.parameters(), lr=lr, betas=(0.9, 0.999))

        optim_cx = optim.Adam(critic_x.parameters(), lr=lr, betas=(0.9, 0.999))
      optim_cz = optim.Adam(critic_z.parameters(), lr=lr, betas=(0.9, 0.999))
      err_loss = torch.nn.MSELoss()

      train_tadgan(encoder, decoder, critic_x, critic_z, n_epochs=params.epochs, params = params)

      if params.hyperbolic: PATH = './models_{}_{}_{}/'.format(params.dataset,str(params.epochs),str(params.lr))+ params.signal
      else: PATH = './models_eucl_{}_{}_{}/'.format(params.dataset,str(params.epochs),str(params.lr))+ params.signal
      if not os.path.isdir(PATH):
        os.makedirs(PATH)

      torch.save(encoder, PATH+'/encoder.pt')
      torch.save(decoder, PATH+'/decoder.pt')
      torch.save(critic_x, PATH+'/critic_x.pt')
      torch.save(critic_z, PATH+'/critic_z.pt')

      if params.dataset in  ['CASAS','new_CASAS']  :
          known_anomalies=[]
      elif params.dataset in ['A1','A2','A3','A4']:
          known_anomalies = pd.read_csv(read_path[:-4]+'_known_anomalies.csv')
      else: 
          known_anomalies = od.load_anomalies(params.signal)

      anomaly_refactor.test_tadgan(test_loader, encoder, decoder, critic_x, critic_z, known_anomalies, read_path=read_path, signal = params.signal, hyperbolic = params.hyperbolic, path=PATH, signal_shape=signal_shape, params=params)
      

    

