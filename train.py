#!usr/bin/bash python
# coding: utf-8
import logging
import os

import geoopt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import models.tadgan as tadgan
from hyperspace.utils import *


def critic_x_iteration(sample, decoder, critic_x, optim_cx, params):
    optim_cx.zero_grad()
    y = sample.view(1, params.batch_size, params.signal_shape)
    valid_x = torch.squeeze(critic_x(y))

    # The sampled z are the anomalous points - points deviating from actual distribution of z (obtained through encoding x)
    z = torch.Tensor(
        np.random.normal(size=(1, params.batch_size, params.latent_space_dim))
    ).cuda()
    if decoder.hyperbolic:
        # hyperspace not used with the critics at the moment
        x_, eucl = decoder(z.cuda())
        # x_ = gmath.logmap0(x_, k=torch.tensor(-1.), dim=1).float()
    else:
        x_ = decoder(z.cuda())

    fake_x = torch.squeeze(critic_x(x_))

    critic_score_valid_x = torch.mean(
        -torch.ones(valid_x.shape).cuda() * valid_x
    )  # Wasserstein Loss
    critic_score_fake_x = torch.mean(
        torch.ones(fake_x.shape).cuda() * fake_x
    )  # Wasserstein Loss

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
    gradients = torch_grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size_, -1)
    # self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)

    # Return gradient penalty
    gp_loss = ((gradients_norm - 1) ** 2).mean()

    #################################################

    # Critic has to maximize Cx(Valid X) - Cx(Fake X).
    # Maximizing the above is same as minimizing the negative.
    wl = critic_score_fake_x + critic_score_valid_x
    loss = wl + 10 * gp_loss
    loss.backward(retain_graph=True)
    optim_cx.step()

    return loss


def critic_z_iteration(sample, encoder, critic_z, optim_cz, params):
    optim_cz.zero_grad()

    x = sample.view(1, params.batch_size, params.signal_shape)
    z_ = encoder(x)

    fake_z = torch.squeeze(critic_z(z_.cuda()))
    critic_score_fake_z = torch.mean(
        torch.ones(fake_z.shape).cuda() * fake_z
    )  # Wasserstein Loss

    z = torch.Tensor(
        np.random.normal(size=(1, params.batch_size, params.latent_space_dim))
    ).cuda()
    valid_z = torch.squeeze(critic_z(z))
    critic_score_valid_z = torch.mean(
        -torch.ones(fake_z.shape).cuda() * valid_z
    )  # Wasserstein Loss

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
    gradients = torch_grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size_, -1)
    # self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)

    # Return gradient penalty
    gp_loss = ((gradients_norm - 1) ** 2).mean()

    #################################################

    loss = wl + 10 * gp_loss
    loss.backward(retain_graph=True)
    optim_cz.step()

    return loss


def decoder_iteration(
    sample,
    encoder,
    decoder,
    critic_x,
    critic_z,
    optim_dec,
    params,
    err_loss=nn.MSELoss(),
):
    optim_dec.zero_grad()

    x_gen = sample.view(1, params.batch_size, params.signal_shape)
    z_gen_ = encoder(x_gen)
    fake_gen_z = critic_z(z_gen_)

    z_gen = torch.Tensor(
        np.random.normal(size=(1, params.batch_size, params.latent_space_dim))
    ).cuda()

    if not decoder.hyperbolic:
        x_gen_ = decoder(z_gen)
    else:
        x_gen_, _ = decoder(z_gen)

    fake_gen_x = critic_x(x_gen_)
    critic_score_fake_gen_x = torch.mean(
        -torch.ones(fake_gen_x.shape).cuda() * fake_gen_x
    )
    critic_score_fake_gen_z = torch.mean(
        -torch.ones(fake_gen_z.shape).cuda() * fake_gen_z
    )

    if decoder.hyperbolic:
        x_gen_rec, eucl = decoder(z_gen_)
        hyper_x = decoder.hyperbolic_linear(x_gen.view(-1, params.signal_shape))

        sqdist = torch.sum((x_gen_rec - hyper_x) ** 2, dim=-1)
        squnorm = torch.sum(x_gen_rec**2, dim=-1)
        sqvnorm = torch.sum(hyper_x**2, dim=-1)
        x_temp = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 1e-7
        dist = torch.acosh(x_temp)

        hyper_loss = torch.div(torch.sum(dist), params.batch_size)
        mse = torch.Tensor([0])
        loss_dec = 10 * hyper_loss + critic_score_fake_gen_x + critic_score_fake_gen_z

        loss_dec.backward(retain_graph=True)
        optim_dec.step()

        return loss_dec, hyper_loss, mse

    else:
        x_gen_rec = decoder(z_gen_)
        mse_loss = err_loss(x_gen.float(), x_gen_rec.float())
        loss_dec = 10 * mse_loss + critic_score_fake_gen_x + critic_score_fake_gen_z

        loss_dec.backward()
        optim_dec.step()

        return loss_dec, 0, mse_loss


def train_tadgan(
    train_loader,
    encoder,
    decoder,
    critic_x,
    critic_z,
    n_epochs=2000,
    params=[],
    path="",
):
    logging.debug("Starting training")
    cx_epoch_loss = list()
    cz_epoch_loss = list()
    encoder_epoch_loss = list()
    decoder_epoch_loss = list()
    hyp_dec_loss = list()
    eucl_dec_loss = list()

    """
    INIT OPTIMIZERS
    """
    # optim_enc = optim.Adam(encoder.parameters(), lr=params.lr, betas=(0.9, 0.999))
    optim_cx = optim.Adam(critic_x.parameters(), lr=params.lr, betas=(0.9, 0.999))
    optim_cz = optim.Adam(critic_z.parameters(), lr=params.lr, betas=(0.9, 0.999))
    # optim_dec = optim.Adam(decoder.parameters(), lr=params.lr, betas=(0.9, 0.999))
    optim_dec = optim.Adam(
        list(decoder.parameters()) + list(encoder.parameters()),
        lr=params.lr,
        betas=(0.9, 0.999),
    )
    if params.hyperbolic:
        optim_dec = geoopt.optim.RiemannianAdam(
            list(decoder.parameters()) + list(encoder.parameters()),
            lr=params.lr,
            weight_decay=1e-5,
            stabilize=10,
        )

    actual_epoch = 0

    if params.resume:
        n_epochs = n_epochs - params.resume_epoch
        actual_epoch = params.resume_epoch + 1

    """
    TRAINING LOOP
    """
    for epoch in range(n_epochs):
        logging.debug("Epoch {}".format(epoch))
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
                loss = critic_x_iteration(
                    sample.cuda(), decoder, critic_x, optim_cx, params
                )
                cx_loss.append(loss)

                loss = critic_z_iteration(
                    sample.cuda(), encoder, critic_z, optim_cz, params
                )
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

        logging.debug("Critic training done in epoch {}".format(epoch))
        encoder_loss = list()
        decoder_loss = list()
        hyp_loss = list()
        mse_losss = list()
        for batch, sample in enumerate(train_loader):
            # enc_loss = encoder_iteration(sample.cuda())
            dec_loss, hyper_loss, mse_loss = decoder_iteration(
                sample.cuda(), encoder, decoder, critic_x, critic_z, optim_dec, params
            )
            # encoder_loss.append(enc_loss)
            decoder_loss.append(dec_loss)
            if params.hyperbolic:
                hyp_loss.append(hyper_loss.float())  # type: ignore
            mse_losss.append(mse_loss.float())

        cx_epoch_loss.append(torch.mean(torch.tensor(cx_nc_loss)))
        cz_epoch_loss.append(torch.mean(torch.tensor(cz_nc_loss)))
        # encoder_epoch_loss.append(torch.mean(torch.tensor(encoder_loss)))
        decoder_epoch_loss.append(torch.mean(torch.tensor(decoder_loss)))

        if params.hyperbolic:
            hyp_dec_loss.append(torch.mean(torch.tensor(hyp_loss)))
        eucl_dec_loss.append(torch.mean(torch.tensor(mse_losss)))

        print("Encoder decoder training done in epoch {}".format(epoch))
        if params.hyperbolic:
            print("Hyperbolic loss {}".format(hyp_dec_loss[-1]))
        else:
            print("Eucl mse loss {}".format(eucl_dec_loss[-1]))
        print(
            "critic x loss {:.3f} critic z loss {:.3f} \ndecoder loss {:.3f}\n".format(
                cx_epoch_loss[-1], cz_epoch_loss[-1], decoder_epoch_loss[-1]
            )
        )
        # print('critic x loss {:.3f} critic z loss {:.3f} \n decoder loss {:.3f}\n'.format(cx_epoch_loss[-1], cz_epoch_loss[-1], decoder_epoch_loss[-1]))

        actual_epoch += 1

        if (actual_epoch % 10 == 0) or (actual_epoch == (n_epochs - 1)):
            torch.save(encoder, path + "/encoder_{}.pt".format(actual_epoch))
            torch.save(decoder, path + "/decoder_{}.pt".format(actual_epoch))
            torch.save(critic_x, path + "/critic_x_{}.pt".format(actual_epoch))
            torch.save(critic_z, path + "/critic_z_{}.pt".format(actual_epoch))


def resume_ckpt(params):
    if params.hyperbolic:
        if params.signal == "multivariate":
            PATH = f"./trained_models/models_hyper_{params.dataset}_{str(params.epochs)}_{str(params.lr)}/{params.dataset}"
        else:
            PATH = f"./trained_models/models_hyper_{params.dataset}_{str(params.epochs)}_{str(params.lr)}/{params.dataset}/{params.signal}"
    else:
        if params.signal == "multivariate":
            PATH = f"./trained_models/models_eucl_{params.dataset}_{str(params.epochs)}_{str(params.lr)}/{params.dataset}"
        else:
            PATH = f"./trained_models/models_eucl_{params.dataset}_{str(params.epochs)}_{str(params.lr)}/{params.dataset}/{params.signal}"

    encoder = torch.load(resume_path + "encoder.pt").cuda().train()
    decoder = torch.load(resume_path + "decoder.pt").cuda().train()
    critic_x = torch.load(resume_path + "critic_x.pt").cuda().train()
    critic_z = torch.load(resume_path + "critic_z.pt").cuda().train()
    print("model resumed from {}".format(resume_path))

    return encoder, decoder, critic_x, critic_z


def train(train_loader, params, config_path):
    """
    MODEL INITIALIZATION
    """
    params.latent_space_dim = 20

    encoder = (
        tadgan.Encoder(params.signal_shape, params.latent_space_dim).cuda().train()
    )
    decoder = (
        tadgan.Decoder(params.signal_shape, params.latent_space_dim, params.hyperbolic)
        .cuda()
        .train()
    )
    critic_x = (
        tadgan.CriticX(params.signal_shape, params.latent_space_dim).cuda().train()
    )
    critic_z = tadgan.CriticZ(params.latent_space_dim).cuda().train()

    if params.hyperbolic:
        if params.signal == "multivariate":
            PATH = f"./trained_models/models_hyper_{params.dataset}_{str(params.epochs)}_{str(params.lr)}/{params.dataset}"
        else:
            PATH = f"./trained_models/models_hyper_{params.dataset}_{str(params.epochs)}_{str(params.lr)}/{params.dataset}/{params.signal}"
    else:
        if params.signal == "multivariate":
            PATH = f"./trained_models/models_eucl_{params.dataset}_{str(params.epochs)}_{str(params.lr)}/{params.dataset}"
        else:
            PATH = f"./trained_models/models_eucl_{params.dataset}_{str(params.epochs)}_{str(params.lr)}/{params.dataset}/{params.signal}"

    if not os.path.isdir(PATH):
        os.makedirs(PATH)

    os.system(f'cp {config_path} {os.path.join(PATH, "config.yaml")}')

    if params.resume:
        encoder, decoder, critic_x, critic_z = resume_ckpt(params)

    """
    TRAIN
    """
    train_tadgan(
        train_loader,
        encoder,
        decoder,
        critic_x,
        critic_z,
        n_epochs=params.epochs,
        params=params,
        path=PATH,
    )

    torch.save(encoder, PATH + "/encoder.pt")
    torch.save(decoder, PATH + "/decoder.pt")
    torch.save(critic_x, PATH + "/critic_x.pt")
    torch.save(critic_z, PATH + "/critic_z.pt")

    return encoder, decoder, critic_x, critic_z, PATH
