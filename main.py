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

import anomaly_detection

from utils import parse_args
import data as od
from dataloader import SignalDataset
from dataloader_multivariate import CasasDataset

import geoopt
import geoopt.manifolds.stereographic.math as gmath
from hyperspace.utils import *

from torch.autograd import grad as torch_grad


def critic_x_iteration(sample):
    optim_cx.zero_grad()
    y = sample.view(sequence_shape, batch_size, signal_shape)
    valid_x = torch.squeeze(critic_x(y))

    #The sampled z are the anomalous points - points deviating from actual distribution of z (obtained through encoding x)
    z = torch.Tensor(np.random.normal(size=(1,batch_size, latent_space_dim))).cuda()
    if decoder.hyperbolic:
      #hyperspace not used with the critics at the moment
      x_,eucl = decoder(z.cuda())
      # x_ = gmath.logmap0(x_, k=torch.tensor(-1.), dim=1).float()
    else: x_ = decoder(z.cuda())
    
    fake_x = torch.squeeze(critic_x(x_))

    critic_score_valid_x = torch.mean(-torch.ones(valid_x.shape).cuda() * valid_x) #Wasserstein Loss
    critic_score_fake_x = torch.mean(torch.ones(fake_x.shape).cuda() * fake_x)  #Wasserstein Loss

    # alpha = torch.rand(y.shape).cuda()
    # ix = Variable(alpha * y + (1 - alpha) * x_) #Random Weighted Average
    # ix.requires_grad_(True)
    # v_ix = critic_x(ix)
    # v_ix.mean().backward()
    # gradients = ix.grad
    # #Gradient Penalty Loss
    # gr_sqr = torch.square(gradients)
    # gr_sqr_sum = torch.sum(gr_sqr,axis=2)
    # gr_l2_norm = torch.sqrt(gr_sqr_sum)

    # gp_loss = torch.mean(torch.square(gr_l2_norm))

    #################################################

    real_data = y
    generated_data = x_
    
    batch_size_ = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(y.shape)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda()
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = critic_x(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size_, -1)
    # self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    gp_loss = ((gradients_norm - 1) ** 2).mean()

    #################################################
 
    #Critic has to maximize Cx(Valid X) - Cx(Fake X).
    #Maximizing the above is same as minimizing the negative.
    wl = critic_score_fake_x + critic_score_valid_x
    loss = wl + 10*gp_loss
    loss.backward(retain_graph=True)
    optim_cx.step()

    return loss


def critic_z_iteration(sample):
    optim_cz.zero_grad()

    x = sample.view(sequence_shape, batch_size, signal_shape)
    z_ = encoder(x)
    
    fake_z = torch.squeeze(critic_z(z_.cuda()))
    critic_score_fake_z = torch.mean(torch.ones(fake_z.shape).cuda() * fake_z) #Wasserstein Loss

    z = torch.Tensor(np.random.normal(size=(1,batch_size, latent_space_dim))).cuda()
    valid_z = torch.squeeze(critic_z(z))
    critic_score_valid_z = torch.mean(-torch.ones(fake_z.shape).cuda() * valid_z) #Wasserstein Loss

    wl = critic_score_fake_z + critic_score_valid_z

    # alpha = torch.rand(z.shape).cuda()
    # iz = Variable(alpha * z + (1 - alpha) * z_) #Random Weighted Average
    # iz.requires_grad_(True)
    # v_iz = critic_z(iz)
    # v_iz.mean().backward()
    # gradients = iz.grad

    # gr_sqr = torch.square(gradients)
    # gr_sqr_sum = torch.sum(gr_sqr,axis=2)
    # gr_l2_norm = torch.sqrt(gr_sqr_sum)

    # gp_loss = torch.mean(torch.square(gr_l2_norm))

    #################################################

    real_data = z
    generated_data = z_
    
    batch_size_ = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(z.shape)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda()
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = critic_z(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size_, -1)
    # self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    gp_loss = ((gradients_norm - 1) ** 2).mean()

    #################################################

    loss = wl + 10*gp_loss
    loss.backward(retain_graph=True)
    optim_cz.step()

    return loss




def decoder_iteration(sample):
    optim_dec.zero_grad()

    x_gen = sample.view(sequence_shape, batch_size, signal_shape)
    z_gen_ = encoder(x_gen)
    fake_gen_z = critic_z(z_gen_)

    z_gen = torch.Tensor(np.random.normal(size=(1,batch_size, latent_space_dim))).cuda()
    if not decoder.hyperbolic:
      x_gen_ = decoder(z_gen)
    else: 
      x_gen_, _ = decoder(z_gen)
    fake_gen_x = critic_x(x_gen_)
    critic_score_fake_gen_x = torch.mean(-torch.ones(fake_gen_x.shape).cuda() * fake_gen_x)
    critic_score_fake_gen_z = torch.mean(-torch.ones(fake_gen_z.shape).cuda() * fake_gen_z)

    if decoder.hyperbolic:
      x_gen_rec, eucl = decoder(z_gen_)
      hyper_x = decoder.hyperbolic_linear(x_gen.view(-1,signal_shape))

      sqdist = torch.sum((x_gen_rec - hyper_x) ** 2, dim=-1)
      squnorm = torch.sum(x_gen_rec ** 2, dim=-1)
      sqvnorm = torch.sum(hyper_x ** 2, dim=-1)
      x_temp = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 1e-7
      dist = torch.acosh(x_temp)

      hyper_loss = torch.div(torch.sum(dist),batch_size)
      mse=torch.Tensor([0])
      loss_dec = 10*hyper_loss + critic_score_fake_gen_x + critic_score_fake_gen_z 

      loss_dec.backward(retain_graph=True)
      optim_dec.step()

      return loss_dec,hyper_loss,mse
      
    else: 
      x_gen_rec = decoder(z_gen_)
      mse_loss = err_loss(x_gen.float(), x_gen_rec.float())
      loss_dec = 10*mse_loss + critic_score_fake_gen_x + critic_score_fake_gen_z 

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

    if params.resume:
      n_epochs = n_epochs - params.resume_epoch
      start_epoch = params.resume_epoch+1
    else: start_epoch=0

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
            
            # enc_loss = encoder_iteration(sample.cuda())
            dec_loss,hyper_loss,mse_loss = decoder_iteration(sample.cuda())
            # encoder_loss.append(enc_loss)
            decoder_loss.append(dec_loss)
            if params.hyperbolic:
              hyp_loss.append(hyper_loss.float())
            mse_losss.append(mse_loss.float())

        cx_epoch_loss.append(torch.mean(torch.tensor(cx_nc_loss)))
        cz_epoch_loss.append(torch.mean(torch.tensor(cz_nc_loss)))
        # encoder_epoch_loss.append(torch.mean(torch.tensor(encoder_loss)))
        decoder_epoch_loss.append(torch.mean(torch.tensor(decoder_loss)))

        if params.hyperbolic:
          hyp_dec_loss.append(torch.mean(torch.tensor(hyp_loss)))
        eucl_dec_loss.append(torch.mean(torch.tensor(mse_losss)))

      
        print('Encoder decoder training done in epoch {}'.format(epoch))
        if params.hyperbolic: print('Hyperbolic loss {}'.format(hyp_dec_loss[-1]))
        print('Eucl mse loss {}'.format(eucl_dec_loss[-1]))
        print('critic x loss {:.3f} critic z loss {:.3f} \ndecoder loss {:.3f}\n'.format(cx_epoch_loss[-1], cz_epoch_loss[-1], decoder_epoch_loss[-1]))
        # print('critic x loss {:.3f} critic z loss {:.3f} \n decoder loss {:.3f}\n'.format(cx_epoch_loss[-1], cz_epoch_loss[-1], decoder_epoch_loss[-1]))
        
        start_epoch+=1

        if (start_epoch % 10 == 0) or (start_epoch == (n_epochs-1)):
            if params.new_features:
              dataset = params.dataset + '_newfeatures'
            else:
              dataset = params.dataset

            if params.hyperbolic: PATH = './trained_models/models_{}_{}_{}/'.format(dataset,str(params.epochs),str(params.lr))+ params.signal
            else: PATH = './trained_models/models_eucl_{}_{}_{}/'.format(dataset,str(params.epochs),str(params.lr))+ params.signal
            if not os.path.isdir(PATH):
                os.makedirs(PATH)
            
            torch.save(encoder, PATH+'/encoder_{}.pt'.format(start_epoch))
            torch.save(decoder, PATH+'/decoder_{}.pt'.format(start_epoch))
            torch.save(critic_x, PATH+'/critic_x_{}.pt'.format(start_epoch))
            torch.save(critic_z, PATH+'/critic_z_{}.pt'.format(start_epoch))




if __name__ == "__main__":
    params = parse_args()
    print('dataset: {}, signal: {}'.format(params.dataset, params.signal))

    '''
    DATASET SELECTION    
    '''
    # original CASAS dataset (test==train).  
    if params.dataset == 'CASAS_':
        path = '/content/drive/MyDrive/ASSISTED LIVING (e-linus)/Data&Experiments/DianeCookCASAS/'
        train_dataset = CasasDataset(seq_path=path+'sequences_2week_{}.pt'.format(params.signal), 
                                     gt_path=path+'ground_truth_2week_{}.pt'.format(params.signal), split=params.split)
        test_dataset = CasasDataset(seq_path=path+'sequences_2week_{}.pt'.format(params.signal), 
                                     gt_path=path+'ground_truth_2week_{}.pt'.format(params.signal), test=True)

    # new_CASAS is the dataset proposed for CVPR '21 where the train&test splits have been created 
    elif params.dataset == 'new_CASAS':
        path = './data/CASAS/new_dataset/'
        train_dataset = CasasDataset(seq_path=path+params.signal, 
                                     gt_path=path+params.signal, split=params.split, dataset=params.dataset)
        test_dataset = CasasDataset(seq_path=path+params.signal, 
                                     gt_path=path+params.signal, test=True, dataset=params.dataset)
        read_path = ''

    elif params.dataset in ['CASAS','ELINUS','eHealth']:
        if not params.new_features:
          seq_path = "./data/DATASETS/{}/normal_sequences.pt".format(params.dataset)
          seq_path_test = "./data/DATASETS/{}/POINTS/{}/{}_sequences_id{}.pt".format(params.dataset, params.signal, params.signal, params.id)
          gt_path = "./data/DATASETS/{}/POINTS/{}/{}_groundtruth_id{}.pt".format(params.dataset, params.signal, params.signal,params.id)
        else:
          seq_path = "./data/DATASETS/{}/normal_sequences_newfeatures.pt".format(params.dataset)
          seq_path_test = "./data/DATASETS/{}/POINTS_NEWFEATURES/{}_sequences_newfeatures.pt".format(params.dataset, params.signal, params.signal)
          gt_path = "./data/DATASETS/{}/POINTS_NEWFEATURES/{}_groundtruth_newfeatures.pt".format(params.dataset, params.signal, params.signal)

        train_dataset = CasasDataset(seq_path=seq_path, 
                                     gt_path=gt_path, split=params.split, dataset=params.dataset)
        test_dataset = CasasDataset(seq_path=seq_path_test, 
                                     gt_path=gt_path, test=True, dataset=params.dataset)
        read_path = ''


    else:
        # univariate dataset with train=test 
        if params.unique_dataset:
            train_data = od.load_signal(params.signal)
            test_data = od.load_signal(params.signal)
            train_dataset = SignalDataset(path='./data/{}.csv'.format(params.signal),interval=params.interval)
            test_dataset = SignalDataset(path='./data/{}.csv'.format(params.signal),test=True,interval=params.interval)
            read_path='./data/{}.csv'.format(params.signal)

        # YAHOO dataset
        elif params.dataset in ['A1','A2','A3','A4']:
            train_dataset = SignalDataset(path='./data/YAHOO/{}Benchmark/{}.csv'.format(params.dataset,params.signal),interval=1,yahoo=True)
            test_dataset = SignalDataset(path='./data/YAHOO/{}Benchmark/{}.csv'.format(params.dataset,params.signal),test=True,interval=1,yahoo=True)
            read_path = './data/YAHOO/{}Benchmark/{}.csv'.format(params.dataset,params.signal)
            
        # univariate dataset with train!=test
        else:
            train_data = od.load_signal(params.signal +'-train')
            test_data = od.load_signal(params.signal +'-test')
            train_dataset = SignalDataset(path='./data/{}-train.csv'.format(params.signal),interval=params.interval)
            test_dataset = SignalDataset(path='./data/{}-test.csv'.format(params.signal),interval=params.interval,test=True)
            read_path='./data/{}-test.csv'.format(params.signal)

    batch_size = params.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True,shuffle=True,num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle=False,num_workers=2)


    '''
    MODEL INITIALIZATION
    '''
    lr = params.lr

    signal_shape = 1
    latent_space_dim = 20
    signal_shape = params.signal_shape
    sequence_shape = 1


    encoder = model.Encoder(signal_shape, latent_space_dim, params.hyperbolic, sequence_shape=sequence_shape).cuda().train()
    decoder = model.Decoder(signal_shape, latent_space_dim, params.hyperbolic).cuda().train()
    critic_x = model.CriticX(signal_shape, latent_space_dim,sequence_shape=sequence_shape).cuda().train()
    critic_z = model.CriticZ(latent_space_dim).cuda().train()
    
    if params.new_features:
      dataset = params.dataset + '_newfeatures'
    else:
      dataset = params.dataset

    if params.hyperbolic: PATH = './trained_models/models_{}_{}_{}/'.format(dataset,str(params.epochs),str(params.lr))+ params.signal
    else: PATH = './trained_models/models_eucl_{}_{}_{}/'.format(dataset,str(params.epochs),str(params.lr))+ params.signal
    if not os.path.isdir(PATH):
        os.makedirs(PATH)

    if params.resume:
      if params.hyperbolic:
        resume_path = './trained_models/models_{}_{}/{}/'.format(params.resume_epoch, params.lr, params.signal)
      else:
        resume_path = './trained_models/models_eucl_{}_{}/{}/'.format(params.resume_epoch, params.lr, params.signal)
      encoder = torch.load(resume_path+'encoder.pt').cuda().train()
      decoder = torch.load(resume_path+'decoder.pt').cuda().train()
      critic_x = torch.load(resume_path+'critic_x.pt').cuda().train()
      critic_z = torch.load(resume_path+'critic_z.pt').cuda().train()
      print('model resumed from {}'.format(resume_path))

    optim_enc = optim.Adam(encoder.parameters(), lr=lr, betas=(0.9, 0.999))
    optim_cx = optim.Adam(critic_x.parameters(), lr=lr, betas=(0.9, 0.999))
    optim_cz = optim.Adam(critic_z.parameters(), lr=lr, betas=(0.9, 0.999))
    optim_dec = optim.Adam(decoder.parameters(), lr=lr, betas=(0.9, 0.999))

    if params.hyperbolic:
        optim_dec = geoopt.optim.RiemannianAdam(decoder.parameters(), lr=lr, weight_decay=1e-5, stabilize=10)

    err_loss = torch.nn.MSELoss()

    '''
    TRAIN
    '''
    train_tadgan(encoder, decoder, critic_x, critic_z, n_epochs=params.epochs, params = params)

    torch.save(encoder, PATH+'/encoder.pt')
    torch.save(decoder, PATH+'/decoder.pt')
    torch.save(critic_x, PATH+'/critic_x.pt')
    torch.save(critic_z, PATH+'/critic_z.pt')

    '''
    ANOMALY DETECTOR
    '''
    # load ground truth anomalies
    if params.dataset in  ['CASAS','ELINUS','eHealth','new_CASAS']  :
          known_anomalies=[]
    elif params.dataset in ['A1','A2','A3','A4']:
          known_anomalies = pd.read_csv(read_path[:-4]+'_known_anomalies.csv')
    else: 
          known_anomalies = od.load_anomalies(params.signal)

    anomaly_detection.test_tadgan(test_loader, encoder, decoder, critic_x, critic_z, known_anomalies, read_path=read_path, signal = params.signal, hyperbolic = params.hyperbolic, path=PATH, signal_shape=signal_shape, params=params)
      