import numpy as np
import os
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset'     , default='MSL',        help='dataset to be loaded')
    parser.add_argument('--signal'     , default='C-2',        help='signal to be loaded')
    parser.add_argument('--epochs', default=70, type=int,help ='number of epochs for the train')
    parser.add_argument('--model'       , default='lstm',      help='model: lstm / tadgan')
    parser.add_argument('--hyperbolic'       , action='store_true', help='use hyperbolic space')
    parser.add_argument('--signal_shape', default=100, type=int,help ='shape of the window')
    parser.add_argument('--id', default=1, type=int,help ='id that tadgan has to consider [Dario\'s dataset only]')
    parser.add_argument('--lr', default=0.005, type=float,help ='learning rate')
    parser.add_argument('--split', default=1, type=int,help ='which split to use for train/test of the CASAS dataset')
    parser.add_argument('--batch_size', default=64, type=int,help ='number of samples per batch')
    parser.add_argument('--verbose', action='store_true',help ='print all the information')
    parser.add_argument('--save_result', action='store_true',      help='save the output of confusion matrix at filename')
    parser.add_argument('--filename'       , default='',      help='name of the file with results')
    parser.add_argument('--rec_error'       , default='dtw',      help='name of the file with results')
    parser.add_argument('--combination'       , default='mult',      help='name of the file with results')
    parser.add_argument('--interval'       , default=21600, type=int,     help='used in the preprocessing phase')
    parser.add_argument('--unique_dataset', action='store_true',      help='if train and test are different')
    parser.add_argument('--resume', action='store_true',help ='true if you want to resume model')
    parser.add_argument('--resume_epoch', default=10, type=int,help ='epoch you want to resume')
    parser.add_argument('--load', action='store_true',help ='enable load saved pickles')
    parser.add_argument('--new_features', action='store_true',help ='ony for the Dario\'s datasets')
    parser.add_argument('--config '       , default='dtw',      help='name of the file with results')

    return parser.parse_args()

def unroll_ts(y_hat):
    predictions = list()
    pred_length = y_hat.shape[1]
    num_errors = y_hat.shape[1] + (y_hat.shape[0] - 1)

    for i in range(num_errors):
            intermediate = []

            for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
                intermediate.append(y_hat[i - j, j])

            if intermediate:
                predictions.append(np.median(np.asarray(intermediate)))

    return np.asarray(predictions[pred_length-1:])

def convert_date(timelist):
    converted = list()
    for x in timelist:
        converted.append(datetime.fromtimestamp(x))
    return converted

def convert_date_single(x):
    return datetime.fromtimestamp(x)

def plot_ts(X, labels=None,num='1'):
    fig = plt.figure(figsize=(30, 6))
    ax = fig.add_subplot(111)
    
    if not isinstance(X, list):
        X = [X]
  
    for x in X:
        t = range(len(x))
        plt.plot(t, x)
    
    plt.title('NYC Taxi Demand', size=34)
    plt.ylabel('# passengers', size=30)
    plt.xlabel('Time', size=30)
    plt.xticks(size=26)
    plt.yticks(size=26)
    plt.xlim([t[0], t[-1]])
    
    if labels:
        plt.legend(labels=labels, loc=1, prop={'size': 26})
    fig.savefig('signals_{}.jpg'.format(num))
    plt.show()

def plot_error(X):
    plt.figure(figsize = (30, 6))
    gs1 = gridspec.GridSpec(3, 1)
    gs1.update(wspace=0.025, hspace=0.05) 

    i = 0
    for x in X:
        if len(x) == 2:
            ax1 = plt.subplot(gs1[i:i+2])
            for line in x:
                t = range(len(line))
                ax1.plot(t, line)
            i+=1
        else:
            ax1 = plt.subplot(gs1[i])
            t = range(len(line))
            ax1.plot(t, x, color='tab:red')

        i+=1
        plt.xlim(t[0], t[-1])
        plt.yticks(size=22)
        plt.axis('on')
        ax1.set_xticklabels([])

    plt.show()


def plot(dfs, anomalies=[], signal = 'NYC Taxi Demand', path=''):
    """ Line plot for time series.
    
    This function plots time series and highlights anomalous regions.
    The first anomaly in anomalies is considered the ground truth.
    
    Args:
        dfs (list or `pd.DataFrame`): List of time series in `pd.DataFrame`.
            Or a single dataframe. All dataframes must have the same shape.
        anomalies (list): List of anomalies in tuple format.
    """    
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        
    if not isinstance(anomalies, list):
        anomalies = [anomalies]
        
    df = dfs[0]
    time = convert_date(df['timestamp'])
    months = mdates.MonthLocator()  # every month
    days = mdates.DayLocator() # every day

    month_fmt = mdates.DateFormatter('%b')

    fig = plt.figure(figsize=(30, 6))
    ax = fig.add_subplot(111)

    for df in dfs:
        plt.plot(time, df['value'])

    colors = ['red'] + ['green'] * (len(anomalies) - 1)
    for i, anomaly in enumerate(anomalies):
        if not isinstance(anomaly, list):
            anomaly = list(anomaly[['start', 'end']].itertuples(index=False))
        
        for _, anom in enumerate(anomaly):
            t1 = convert_date_single(anom[0])
            t2 = convert_date_single(anom[1])
            plt.axvspan(t1, t2, color=colors[i], alpha=0.2)

    plt.title(signal, size=34)
    # plt.ylabel('# passengers', size=30)
    plt.xlabel('Time', size=30)
    plt.xticks(size=26)
    plt.yticks(size=26)
    plt.xlim([time[0], time[-1]])

    # format xticks
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(month_fmt)
    ax.xaxis.set_minor_locator(days)
    
    # format yticks
    # ylabels = ['{:,.0f}'.format(x) + 'K' for x in ax.get_yticks()/1000]
    # ax.set_yticklabels(ylabels)
    print(path+'anomalies.jpg')
    fig.savefig(path+'anomalies.jpg')
    plt.show()
    
    
def plot_rws(X, window=100, k=5, lim=1000):
    shift = 75
    X = X[window:]
    t = range(len(X))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    
    num_figs = int(np.ceil(k / 5)) + 1
    fig = plt.figure(figsize=(15, num_figs * 2))
    
    j = 0
    ax = fig.add_subplot(num_figs, 5, j+1)
    idx = t[j: window + j]
    ax.plot(idx, X[j], lw=2, color=colors[j])
    plt.title("window %d" % j, size=16)
    plt.ylim([-1, 1])
    
    j = 1
    ax = fig.add_subplot(num_figs, 5, j+1)
    idx = t[j: window + j]
    ax.plot(idx, X[j], lw=2, color=colors[j])
    ax.set_yticklabels([])
    plt.title("window %d" % j, size=16)
    plt.ylim([-1, 1])
        
    for i in range(2, k):
        j = i * shift
        idx = t[j: window + j]
        
        ax = fig.add_subplot(num_figs, 5, i+1)
        ax.plot(idx, X[j], lw=2, color=colors[i+1])
        ax.set_yticklabels([])
        plt.title("window %d" % j, size=16)
        plt.ylim([-1, 1])
    
    plt.tight_layout()
    plt.show()

