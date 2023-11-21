import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

import utils.data as od
from utils.anomaly_detection_utils import (
    multivariate_anomaly_detection,
    univariate_anomaly_detection,
)

warnings.filterwarnings("ignore")


def test_tadgan(
    test_loader,
    encoder,
    decoder,
    critic_x,
    read_path="",
    signal="",
    path="",
    signal_shape=100,
    params=[],
):
    # load ground truth anomalies
    if params.signal == "multivariate":
        known_anomalies = []
    elif params.dataset in ["A1", "A2", "A3", "A4"]:  # YAHOO dataset
        known_anomalies = pd.read_csv(read_path[:-4] + "_known_anomalies.csv")
    else:
        known_anomalies = od.load_anomalies(params.signal)

    recons_signal = []
    true_signal = []
    critic_score = list()
    hyper_real = []
    eucl_recons = []
    path += "/"

    decoder.eval()
    encoder.eval()
    critic_x.eval()
    rec_error_type = params.rec_error
    combination = params.combination

    # Load saved tensors to avoid recomputing the embeddings
    if (
        params.load
        and (os.path.exists(path + "critic_score.pt"))
        and (os.path.exists(path + "recons_signal.pt"))
    ):
        recons_signal = torch.load(path + "recons_signal.pt")
        true_signal = torch.load(path + "gt_signal.pt")
        critic_score = torch.load(path + "critic_score.pt")
        true_index = torch.load(path + "true_index.pt")

    else:
        """
        TESTING LOOP
        """
        for batch, (sample, index, y, y_index, x_index) in enumerate(test_loader):
            x = encoder(sample.float().cuda())

            if decoder.hyperbolic:
                hyper, eucl = decoder(x)
                hyper_x = decoder.hyperbolic_linear(
                    sample.view(-1, signal_shape).float().cuda()
                )

                if sample.shape[0] == 1:
                    recons_signal.append(
                        torch.squeeze(hyper).cpu().detach().numpy().reshape(1, -1)
                    )
                    eucl_recons.append(
                        torch.squeeze(eucl).cpu().detach().numpy().reshape(1, -1)
                    )
                    hyper_real.append(
                        torch.squeeze(hyper_x).cpu().detach().numpy().reshape(1, -1)
                    )
                    critic_score.extend(
                        critic_x(sample.cuda()).cpu().detach().numpy().reshape(-1)
                    )
                else:
                    recons_signal.append(torch.squeeze(hyper).cpu().detach().numpy())
                    eucl_recons.append(torch.squeeze(eucl).cpu().detach().numpy())
                    hyper_real.append(torch.squeeze(hyper_x).cpu().detach().numpy())
                    critic_score.extend(
                        torch.squeeze(critic_x(sample.cuda())).cpu().detach().numpy()
                    )
            else:
                reconstructed_signal = decoder(x)
                if sample.shape[0] == 1:
                    recons_signal.append(
                        reconstructed_signal.cpu().detach().numpy().reshape(1, -1)
                    )
                    critic_score.extend(
                        critic_x(sample.cuda()).cpu().detach().numpy().reshape(-1)
                    )
                else:
                    recons_signal.append(
                        torch.squeeze(reconstructed_signal).cpu().detach().numpy()
                    )
                    critic_score.extend(
                        torch.squeeze(critic_x(sample.cuda())).cpu().detach().numpy()
                    )

            true_signal.append(sample.numpy())

        # save tensors for visualizations and post-processing
        recons_signal = np.concatenate(recons_signal)
        gt_signal = np.concatenate(true_signal)
        torch.save(recons_signal, path + "recons_signal.pt")
        torch.save(gt_signal, path + "gt_signal.pt")
        true_signal = np.concatenate(true_signal)
        torch.save(critic_score, path + "critic_score.pt")
        try:
            torch.save(index[0], path + "true_index.pt")  # type: ignore
        except:
            pass

        if decoder.hyperbolic:
            true_signal = np.concatenate(hyper_real)
            eucl_recons = np.concatenate(eucl_recons)
            torch.save(eucl_recons, path + "eucl_recons.pt")
            torch.save(true_signal, path + "real_hyper.pt")
        if not params.signal == "multivariate":
            x_index = index[0]  # type: ignore
            true_index = x_index

    if params.signal == "multivariate":
        multivariate_anomaly_detection(
            recons_signal, true_signal, params, combination, critic_score, path
        )

    else:
        univariate_anomaly_detection(
            recons_signal,
            true_signal,
            params,
            combination,
            critic_score,
            path,
            read_path,
            rec_error_type,
            true_index,
            known_anomalies,
            signal,
            signal_shape,
        )


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

    train_dataset, test_dataset, read_path = od.dataset_selection(params)

    batch_size = params.batch_size
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=6)  # type: ignore

    dataset = params.dataset

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

    if (params.dataset in ["CASAS", "ELINUS", "eHealth"]) and (not params.new_features):
        PATH += "_id{}/".format(params.id)
    if not os.path.isdir(PATH):
        os.makedirs(PATH)

    if params.dataset in ["CASAS", "ELINUS", "eHealth"]:
        if not params.hyperbolic:
            load_path = "./trained_models/models_eucl_{}_{}_{}/BedDuration".format(
                dataset, str(params.epochs), str(params.lr)
            )
        else:
            load_path = "./trained_models/models_{}_{}_{}/BedDuration".format(
                dataset, str(params.epochs), str(params.lr)
            )

    else:
        load_path = PATH

    if params.resume:
        # if needed to test a specific epoch
        print("resuming epoch: {}".format(params.resume_epoch))
        encoder = torch.load(
            load_path + "/encoder_{}.pt".format(params.resume_epoch)
        ).cuda()
        decoder = torch.load(
            load_path + "/decoder_{}.pt".format(params.resume_epoch)
        ).cuda()
        critic_x = torch.load(
            load_path + "/critic_x_{}.pt".format(params.resume_epoch)
        ).cuda()
    else:
        # loading the last epoch
        encoder = torch.load(load_path + "/encoder.pt").cuda()
        decoder = torch.load(load_path + "/decoder.pt").cuda()
        critic_x = torch.load(load_path + "/critic_x.pt").cuda()

    encoder.eval()
    decoder.eval()
    critic_x.eval()

    test_tadgan(
        test_loader,
        encoder,
        decoder,
        critic_x,
        read_path=read_path,
        signal=params.signal,
        path=PATH,
        signal_shape=params.signal_shape,
        params=params,
    )
