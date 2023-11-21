#!usr/bin/bash python
# coding: utf-8

import argparse

import yaml
from torch.utils.data import DataLoader

import anomaly_detection
import utils.data as od
from hyperspace.utils import *
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HypAD")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        default="/your_default_config_file_path",
    )

    params = parser.parse_args()
    config_path = params.config
    params = yaml.load(open(params.config), Loader=yaml.FullLoader)
    params = argparse.Namespace(**params)

    print("dataset: {}, signal: {}".format(params.dataset, params.signal))
    print(params)

    train_dataset, test_dataset, read_path = od.dataset_selection(params)

    batch_size = params.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=2,
    )

    """
      TRAINING
      """
    encoder, decoder, critic_x, critic_z, path = train(
        train_loader, params, config_path
    )

    """
      ANOMALY DETECTOR
      """
    anomaly_detection.test_tadgan(
        test_loader,
        encoder,
        decoder,
        critic_x,
        read_path=read_path,
        signal=params.signal,
        path=path,
        signal_shape=params.signal_shape,
        params=params,
    )
