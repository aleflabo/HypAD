#!/usr/bin/env python
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
import anomaly_detection_CASAS

from utils import parse_args
import data as od
from dataloader import SignalDataset
from dataloader_multivariate import CasasDataset

from hyperspace.poincare_distance import poincare_distance
import geoopt
import geoopt.manifolds.stereographic.math as gmath
from hyperspace.losses import compute_mask,compute_scores
from hyperspace.utils import *


logging.basicConfig(filename='train.log', level=logging.DEBUG)

def critic_x_iteration(sample):
    optim_cx.zero_grad()
    x = sample.view(sequence_shape, batch_size, signal_shape)
    # x = decoder.hyperbolic_linear(x.view(-1,100))
    # x = gmath.logmap0(x, k=torch.tensor(-1.), dim=1).float()
    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)
    critic_score_valid_x = torch.mean(torch.ones(valid_x.shape).cuda() * valid_x) #Wasserstein Loss

    #The sampled z are the anomalous points - points deviating from actual distribution of z (obtained through encoding x)
    z = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
    if decoder.hyperbolic:
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
    loss.backward()
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
    loss.backward()
    optim_cz.step()

    return loss

def encoder_iteration(sample):
    optim_enc.zero_grad()

    x = sample.view(sequence_shape, batch_size, signal_shape)
    # x = decoder.hyperbolic_linear(x.view(-1,100))
    # x = gmath.logmap0(x, k=torch.tensor(-1.), dim=1).float()
    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)
    critic_score_valid_x = torch.mean(torch.ones(valid_x.shape).cuda() * valid_x) #Wasserstein Loss

    z = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1).cuda()
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
      # hyper_x = decoder.dist(hyper_x)
      # hyper_x = decoder.lin2(decoder.relu(decoder.lin1(hyper_x)))
      # mse_loss = err_loss(hyper_x, gen_x)

      sqdist = torch.sum((gen_x - hyper_x) ** 2, dim=-1)
      squnorm = torch.sum(gen_x ** 2, dim=-1)
      sqvnorm = torch.sum(hyper_x ** 2, dim=-1)
      x_temp = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 1e-7
      # z = torch.sqrt(x_temp ** 2 - 1)
      # dist = torch.log(x_temp + z)
      dist = torch.acosh(x_temp)

      mse_loss = torch.div(torch.sum(dist),batch_size)

      # gen_x_eucl = gmath.logmap0(gen_x, k=torch.tensor(-1.), dim=1).float()
      mse = err_loss(eucl.float(), x.float())
      alpha = 1
      loss_enc = mse + (alpha)*mse_loss + critic_score_valid_x - critic_score_fake_x

      # print(mse_loss)   
      # target, sizes_mask = compute_mask((1,1,1), batch_size)

      # score = compute_scores(gen_x, hyper_x, (1,1,1), batch_size)

      # _, B2, NS, NP, SQ = sizes_mask
      # # score is a 6d tensor: [B, P, SQ, B2, N, SQ]
      # # similarity matrix is computed inside each gpu, thus here B == num_gpu * B2
      # score_flattened = score.contiguous().view(batch_size * NP * SQ, B2 * NS * SQ)

      # target_flattened = target.contiguous().view(batch_size * NP * SQ, B2 * NS * SQ)
      # target_flattened = target_flattened.float().argmax(dim=1)

      # loss = nn.functional.cross_entropy(score_flattened, target_flattened)
      # mse_loss = loss

    else: 
      gen_x = decoder(enc_z)
      # print('x: {}, encoding: {}, generated_x: {}'.format(x.shape, enc_z.shape, gen_x.shape))
      mse = err_loss(x.float(), gen_x.float())
      loss_enc = mse + critic_score_valid_x - critic_score_fake_x

    loss_enc.backward(retain_graph=True)
    optim_enc.step()

    return loss_enc


# def poincare_loss(output, target):
#     loss = torch.mean((output - target)**2)
#     return loss

def decoder_iteration(sample):
    optim_dec.zero_grad()

    x = sample.view(sequence_shape, batch_size, signal_shape)
    if encoder.hyperbolic:
      z, _ = encoder(x)
      # mse_loss = err_loss(x.float(), gen_x.float())
    else: z = encoder(x)
    valid_z = critic_z(z)
    valid_z = torch.squeeze(valid_z)
    critic_score_valid_z = torch.mean(torch.ones(valid_z.shape).cuda() * valid_z)

    z_ = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1).cuda()
    fake_z = critic_z(z_)
    fake_z = torch.squeeze(fake_z)
    critic_score_fake_z = torch.mean(torch.ones(fake_z.shape).cuda() * fake_z)

    if encoder.hyperbolic:
      enc_z, _ = encoder(x)
      # mse_loss = err_loss(x.float(), gen_x.float())
    else: enc_z = encoder(x)
    if decoder.hyperbolic:
      gen_x,eucl = decoder(enc_z)
      hyper_x = decoder.hyperbolic_linear(x.view(-1,signal_shape))
      mse=0

      # target, sizes_mask = compute_mask((1,1,1), batch_size)

      # score = compute_scores(gen_x, hyper_x, (1,1,1), batch_size)


      # _, B2, NS, NP, SQ = sizes_mask
      # # score is a 6d tensor: [B, P, SQ, B2, N, SQ]
      # # similarity matrix is computed inside each gpu, thus here B == num_gpu * B2
      # score_flattened = score.contiguous().view(batch_size * NP * SQ, B2 * NS * SQ)

      # target_flattened = target.contiguous().view(batch_size * NP * SQ, B2 * NS * SQ)
      # target_flattened = target_flattened.float().argmax(dim=1)

      # loss = nn.functional.cross_entropy(score_flattened, target_flattened)
      # # print(loss)
      # # top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1, 3, 5))

      # # results = top1, top3, top5, loss.item(), batch_size
      # mse_loss = loss

      # hyper_x = decoder.dist(hyper_x)
      # hyper_x = decoder.lin2(decoder.relu(decoder.lin1(hyper_x)))
      # mse_loss = err_loss(hyper_x, gen_x)


      sqdist = torch.sum((gen_x - hyper_x) ** 2, dim=-1)
      squnorm = torch.sum(gen_x ** 2, dim=-1)
      sqvnorm = torch.sum(hyper_x ** 2, dim=-1)
      x_temp = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 1e-7
      # z = torch.sqrt(x_temp ** 2 - 1)
      # dist = torch.log(x_temp + z)
      dist = torch.acosh(x_temp)
      # print(dist)

      hyper_loss = torch.div(torch.sum(dist),batch_size)

      # gen_x_eucl = gmath.logmap0(gen_x, k=torch.tensor(-1.), dim=1).float()
      # print(gen_x_eucl.shape,x.shape)
      mse = err_loss(eucl.float(), x.float())
      alpha = 1
      loss_dec = mse + (alpha)*hyper_loss + critic_score_valid_z - critic_score_fake_z
     
      loss_dec.backward(retain_graph=True)
      optim_dec.step()

      return loss_dec,hyper_loss,mse

      
    else: 
      gen_x = decoder(enc_z)
      mse_loss = err_loss(x.float(), gen_x.float())
      loss_dec = mse_loss + critic_score_valid_z - critic_score_fake_z

      loss_dec.backward(retain_graph=True)
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

        for i in range(n_critics):
            cx_loss = list()
            cz_loss = list()

            for batch, sample in enumerate(train_loader):
                # sample = torch.Tensor(sample.numpy().repeat(10,2))

                loss = critic_x_iteration(sample.cuda())
                cx_loss.append(loss)

                loss = critic_z_iteration(sample.cuda())
                cz_loss.append(loss)

            cx_nc_loss.append(torch.mean(torch.tensor(cx_loss)))
            cz_nc_loss.append(torch.mean(torch.tensor(cz_loss)))

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
        if params.hyperbolic:
          hyp_dec_loss.append(torch.mean(torch.tensor(hyp_loss)))
        eucl_dec_loss.append(torch.mean(torch.tensor(mse_losss)))
        logging.debug('Encoder decoder training done in epoch {}'.format(epoch))
        logging.debug('critic x loss {:.3f} critic z loss {:.3f} \nencoder loss {:.3f} decoder loss {:.3f}\n'.format(cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))
        print('Encoder decoder training done in epoch {}'.format(epoch))
        if params.hyperbolic: print('Hyperbolic loss {}'.format(hyp_dec_loss[-1]))
        print('Eucl mse loss {}'.format(eucl_dec_loss[-1]))
        print('critic x loss {:.3f} critic z loss {:.3f} \nencoder loss {:.3f} decoder loss {:.3f}\n'.format(cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))

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
    signal_list = ['heartrate_seconds_merged','minuteStepsNarrow_merged','hourlyCalories_merged','minuteSleep_merged']
    demo=False
    if params.dataset == 'CASAS':
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

    else:
        if params.signal == 'nyc_taxi':
          train_data = od.load_signal('nyc_taxi')
          test_data = od.load_signal('nyc_taxi')

          train_dataset = SignalDataset(path='./data/{}.csv'.format(params.signal),interval=1800)
          test_dataset = SignalDataset(path='./data/{}.csv'.format(params.signal),test=True,interval=1800)
          demo=True
          path = './data/{}.csv'
        elif params.signal in signal_list:
          # train_data = od.load_signal('nyc_taxi')
          # test_data = od.load_signal('nyc_taxi')

          train_dataset = SignalDataset(path='./data/Fitabase Data 4.12.16-5.12.16/{}.csv'.format(params.signal),interval=params.interval,fitbit=True,user_id=params.user_id)
          test_dataset = SignalDataset(path='./data/Fitabase Data 4.12.16-5.12.16/{}.csv'.format(params.signal),test=True,interval=params.interval,fitbit=True,user_id=params.user_id)
          demo=True
          path = './data/Fitabase Data 4.12.16-5.12.16/{}.csv'

        else:
          train_data = od.load_signal(params.signal +'-train')
          test_data = od.load_signal(params.signal +'-test')

          train_dataset = SignalDataset(path='./data/{}-train.csv'.format(params.signal),interval=21600)
          test_dataset = SignalDataset(path='./data/{}-test.csv'.format(params.signal),interval=21600,test=True)
          demo=False
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
      # signal_shape = 150
      sequence_shape = 1
      encoder = model.Encoder(encoder_path, signal_shape, latent_space_dim, params.hyperbolic, sequence_shape=sequence_shape).cuda().train()
      decoder = model.Decoder(decoder_path, signal_shape, latent_space_dim, params.hyperbolic).cuda().train()
      critic_x = model.CriticX(critic_x_path, signal_shape, latent_space_dim,sequence_shape=sequence_shape).cuda().train()
      critic_z = model.CriticZ(critic_z_path, latent_space_dim).cuda().train()

      optim_enc = optim.Adam(encoder.parameters(), lr=lr, betas=(0.9, 0.999))
      optim_cx = optim.Adam(critic_x.parameters(), lr=lr, betas=(0.9, 0.999))

      if params.hyperbolic:
        # optim_enc = geoopt.optim.RiemannianAdam(decoder.parameters(), lr=lr, weight_decay=1e-5, stabilize=10)
        optim_dec = geoopt.optim.RiemannianAdam(decoder.parameters(), lr=lr_hyper, weight_decay=1e-5, stabilize=10)
        # optim_cx = geoopt.optim.RiemannianAdam(critic_x.parameters(), lr=lr, weight_decay=1e-5, stabilize=10)
        # optim_cz = geoopt.optim.RiemannianAdam(critic_z.parameters(), lr=lr, weight_decay=1e-5, stabilize=10)
      else: 
        optim_enc = optim.Adam(encoder.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_dec = optim.Adam(decoder.parameters(), lr=lr, betas=(0.9, 0.999))
        optim_cx = optim.Adam(critic_x.parameters(), lr=lr, betas=(0.9, 0.999))
      optim_cz = optim.Adam(critic_z.parameters(), lr=lr, betas=(0.9, 0.999))
      err_loss = torch.nn.MSELoss()

      train_tadgan(encoder, decoder, critic_x, critic_z, n_epochs=params.epochs, params = params)

      if params.hyperbolic: PATH = './models/'+ params.signal
      else: PATH = './models_eucl/'+ params.signal
      if not os.path.isdir(PATH):
        os.makedirs(PATH)

      torch.save(encoder, PATH+'/encoder.pt')
      torch.save(decoder, PATH+'/decoder.pt')
      torch.save(critic_x, PATH+'/critic_x.pt')
      torch.save(critic_z, PATH+'/critic_z.pt')

      if params.dataset not in  ['CASAS','new_CASAS']  :
        if params.signal in signal_list:
          known_anomalies = pd.DataFrame()
        else:
          known_anomalies = od.load_anomalies(params.signal)
      else: known_anomalies=[]

      if demo:
        anomaly_detection_lstm.test_tadgan(test_loader, encoder, decoder, critic_x, critic_z, known_anomalies, read_path=path.format(params.signal), signal = params.signal, hyperbolic = params.hyperbolic, path=PATH)
      else:
        anomaly_detection_lstm.test_tadgan(test_loader, encoder, decoder, critic_x, critic_z, known_anomalies, read_path='./data/{}-test.csv'.format(params.signal),signal = params.signal, hyperbolic = params.hyperbolic, path=PATH, signal_shape=signal_shape)
      
      # else:
          # anomaly_detection_CASAS.test_tadgan_casas(test_loader, encoder, decoder, critic_x, critic_z, signal = params.signal, hyperbolic = params.hyperbolic, path=PATH, signal_shape=signal_shape)


    

