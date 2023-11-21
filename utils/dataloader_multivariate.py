import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from dateutil.rrule import DAILY, SECONDLY, rrule
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class MultivariateDataset(Dataset):
    def __init__(
        self,
        seq_path=None,
        gt_path=None,
        test=False,
        split=1,
        dataset="CASAS",
    ):
        self.test = test

        if dataset == "CASAS_":
            self.X = torch.load(seq_path)
            self.y = torch.load(gt_path)
            self.X = self.X.reshape(self.X.shape[0] * self.X.shape[1], -1)[4500:]
            self.y = self.y.reshape(self.y.shape[0] * self.y.shape[1], -1)[4500:]

            sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
            train_index, test_index = list(sss.split(self.X, self.y))[split - 1]
            init = np.where(self.y == 1)[0][0] - 1000
            end = np.where(self.y == 1)[0][-1] + 1000
            if self.test:
                print(
                    "total length: {}, test length: {}, train length: {}".format(
                        self.y.shape[0], end - init, self.y.shape[0] - (end - init)
                    )
                )

            if not self.test:
                self.y = self.y[:init]
                self.X = self.X[:init].reshape(-1, 150)

            else:
                self.y = self.y[init:end]
                self.X = self.X[init:end].reshape(-1, 150)

        elif dataset == "new_CASAS":
            if self.test:
                self.X = torch.load(os.path.join(seq_path, "x_test")).reshape(-1, 150)
                self.y = torch.load(os.path.join(seq_path, "y_test"))
                scaler = MinMaxScaler(feature_range=(-1, 1))
                self.X = scaler.fit_transform(self.X)

            else:
                self.X = torch.load(os.path.join(seq_path, "x_train")).reshape(-1, 150)
                self.y = torch.load(os.path.join(seq_path, "y_train"))
                scaler = MinMaxScaler(feature_range=(-1, 1))
                self.X = scaler.fit_transform(self.X)

        elif dataset in ["CASAS", "ELINUS", "eHealth"]:  # test == train
            self.X = torch.load(seq_path).reshape(-1, 150)
            self.y = torch.load(gt_path)
            scaler = MinMaxScaler(feature_range=(-1, 1))
            self.X = scaler.fit_transform(self.X)

        elif dataset == "SWAT":
            if not self.test:
                ## downsampled every 5 seconds as in 'Time Series Anomaly Detection for Cyber-physical Systems via Neural System Identification and Bayesian Filtering'
                min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
                X = pd.read_csv("./data/SWAT/SWaT_train_mine.csv", index_col=0).drop(
                    ["Timestamp", "Normal/Attack"], axis=1
                )
                min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
                imp = SimpleImputer()
                X = imp.fit_transform(X)
                self.X = min_max_scaler.fit_transform(X)
            else:
                min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
                X = pd.read_csv("./data/SWAT/SWaT_test_mine.csv", index_col=0).drop(
                    ["Timestamp", "Normal/Attack", "label"], axis=1
                )
                imp = SimpleImputer()
                X = imp.fit_transform(X)
                self.X = min_max_scaler.fit_transform(X)

        elif dataset == "WADI":
            if not self.test:
                ## downsampled every 5 seconds as in 'Time Series Anomaly Detection for Cyber-physical Systems via Neural System Identification and Bayesian Filtering'
                X = pd.read_csv("./data/WADI_downsampled/WADI_train.csv")
                min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
                imp = SimpleImputer()
                X = imp.fit_transform(X)
                self.X = min_max_scaler.fit_transform(X)
            else:
                min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
                X = pd.read_csv("./data/WADI_downsampled/WADI_test_mine.csv").drop(
                    ["Time", "label"], axis=1
                )
                imp = SimpleImputer()
                X = imp.fit_transform(X)
                self.X = min_max_scaler.fit_transform(X)

        else:
            print("Dataset not supported")
            sys.exit(0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X[idx]
        x = torch.from_numpy(row)

        if self.test:
            return x, [], self.y, [], []
        return x
