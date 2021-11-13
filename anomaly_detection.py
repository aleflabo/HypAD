import numpy as np
from scipy import stats,integrate
import pickle
import torch
import math
import pandas as pd
from pyts.metrics import dtw
from scipy.optimize import fmin
from utils import plot, plot_ts, plot_rws, plot_error, unroll_ts
import matplotlib.pyplot as plt
import data as od
import pandas as pd 
from utils import parse_args
from dataloader import SignalDataset
from torch.utils.data import Dataset, DataLoader
import model
from datetime import datetime
import geoopt.manifolds.stereographic.math as gmath
from numba import jit
from dataloader_multivariate import CasasDataset
from datetime import datetime
from dateutil.rrule import rrule, DAILY,SECONDLY
import os

def test_tadgan(test_loader, encoder, decoder, critic_x, critic_z, known_anomalies, read_path='', signal = '', hyperbolic = False, path='', signal_shape=100, params=''):
    # path = './data'

    recons_signal=[]
    true_signal=[]
    critic_score = list()
    hyper_recons = []
    hyper_recons_z = []
    hyper_real = []
    eucl_recons = []
    path+='/'

    decoder.eval()
    encoder.eval()
    critic_x.eval()
    rec_error_type=params.rec_error
    combination=params.addition_type
    for batch, (sample,index,y,y_index,x_index) in enumerate(test_loader):
      if encoder.hyperbolic:
        x, hyper_z = encoder(sample.float().cuda())
        hyper_recons_z.append(torch.squeeze(hyper_z).cpu().detach().numpy())
      else: x = encoder(sample.float().cuda())
      
      if decoder.hyperbolic:
        hyper, eucl = decoder(x)
        hyper_x = decoder.hyperbolic_linear(sample.view(-1,signal_shape).float().cuda())

        # recons_signal.append(torch.squeeze(hyper).cpu().detach().numpy())
        # eucl_recons.append(torch.squeeze(eucl).cpu().detach().numpy())
        # hyper_real.append(torch.squeeze(hyper_x).cpu().detach().numpy())
        
        if sample.shape[0] == 1:
          recons_signal.append(torch.squeeze(hyper).cpu().detach().numpy().reshape(1,100))
          eucl_recons.append(torch.squeeze(eucl).cpu().detach().numpy().reshape(1,100))
          hyper_real.append(torch.squeeze(hyper_x).cpu().detach().numpy().reshape(1,100))
          critic_score.extend(critic_x(sample.cuda()).cpu().detach().numpy().reshape(-1))
        else: 
          recons_signal.append(torch.squeeze(hyper).cpu().detach().numpy())
          eucl_recons.append(torch.squeeze(eucl).cpu().detach().numpy())
          hyper_real.append(torch.squeeze(hyper_x).cpu().detach().numpy())
          critic_score.extend(torch.squeeze(critic_x(sample.cuda())).cpu().detach().numpy())
      else: 
        reconstructed_signal = decoder(x)
        if sample.shape[0] == 1:
          recons_signal.append(reconstructed_signal.cpu().detach().numpy().reshape(1,100))
          critic_score.extend(critic_x(sample.cuda()).cpu().detach().numpy().reshape(-1))
        else:
          recons_signal.append(torch.squeeze(reconstructed_signal).cpu().detach().numpy())
          critic_score.extend(torch.squeeze(critic_x(sample.cuda())).cpu().detach().numpy())

      true_signal.append(sample.numpy())
        

      # else:
      #   print(sample.shape[0])
    # if len(recons_signal[-1].shape) == 1:
    #   if decoder.hyperbolic:
    #     eucl_recons = np.append(np.concatenate(eucl_recons[:-1]),eucl_recons[-1].reshape(1,-1),axis=0)
    #     recons_signal = np.append(np.concatenate(recons_signal[:-1]),recons_signal[-1].reshape(1,-1),axis=0)
    #     hyper_real = np.append(np.concatenate(hyper_real[:-1]),hyper_real[-1].reshape(1,-1),axis=0)
    #   else:
    #     recons_signal = np.append(np.concatenate(recons_signal[:-1]),recons_signal[-1].reshape(1,-1),axis=0)
    #     # critic_score = np.append(np.concatenate(critic_score[:-1]),critic_score[-1].reshape(1,-1),axis=0)
    #   if encoder.hyperbolic:
    #     hyper_recons_z = np.append(np.concatenate(hyper_recons_z[:-1]),hyper_recons_z[-1].reshape(1,-1),axis=0)

    # else:
    recons_signal = np.concatenate(recons_signal)
    gt_signal = np.concatenate(true_signal)
    torch.save(recons_signal, path+'recons_signal.pt')
    torch.save(gt_signal, path+'gt_signal.pt')
    true_signal = np.concatenate(true_signal)
    torch.save(critic_score, path+'critic_score.pt')
    try:
      torch.save(index[0], path+'true_index.pt')
    except:pass
    if decoder.hyperbolic:
      true_signal = np.concatenate(hyper_real)
      eucl_recons = np.concatenate(eucl_recons)
      torch.save(critic_score, path+'critic_score.pt')
      torch.save(eucl_recons, path+'eucl_recons.pt')
      torch.save(recons_signal, path+'recons_hyper.pt')
      torch.save(true_signal, path+'real_hyper.pt')
    if encoder.hyperbolic:
      hyper_recons_z = np.concatenate(hyper_recons_z)
      torch.save(hyper_recons, path+'hyper_recons_z.pt')

    if params.dataset == 'new_CASAS':
        #just creating some random datetime index for the plots
        random_index = list(rrule(SECONDLY, dtstart=datetime(2012, 11, 24), until=datetime(2012,11,30)))[:recons_signal.shape[0]]
        x_index = np.array(list(map(lambda x: datetime.timestamp(x), random_index)))
        index = np.array(list(map(lambda x: datetime.timestamp(x), random_index)))
        torch.save(x_index, path+'x_index.pt')

        y=y[0]
        y=torch.load('./data/CASAS/new_dataset/{}/y_test'.format(params.signal))
        y = y.reshape(y.shape[0]*y.shape[1], -1)[:x_index.shape[0]]
        if not params.hyperbolic:
          error, true_index, true, pred = score_anomalies(true_signal[:recons_signal.shape[0]].reshape(-1,params.signal_shape,1), 
                                                          recons_signal.reshape(-1,params.signal_shape,1), 
                                                          critic_score[:recons_signal.shape[0]], 
                                                          x_index, 
                                                          rec_error_type=rec_error_type, 
                                                          comb=combination)
          torch.save(true_index, path+'true_index.pt')
          torch.save(error, path+'error.pt')
          intervals = find_anomalies(error[:recons_signal.shape[0]].reshape(-1), true_index, 
                                window_size_portion=0.33, 
                                window_size=1410,
                                window_step_size_portion=None, 
                                fixed_threshold=True,
                                anomaly_padding=50)
        else:
          true_data = torch.Tensor(recons_signal).reshape(-1,params.signal_shape)
          pred_data = torch.Tensor(true_signal).reshape(-1,params.signal_shape)
          sqdist = torch.sum((pred_data - true_data) ** 2, dim=1)
          squnorm = torch.sum(pred_data ** 2, dim=-1)
          sqvnorm = torch.sum(true_data ** 2, dim=-1)
          x_temp = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 1e-7
          dist = torch.acosh(x_temp)
          
          torch.save(x_index, path+'true_index.pt')
          #using poincarÃ¨ distances between gt and preds as errors
          intervals = find_anomalies(dist, x_index, 
                                window_size_portion=0.2, 
                                window_size=1410,
                                window_step_size_portion=None, 
                                fixed_threshold=True,
                                anomaly_padding=200)

        pred_anomalies = pd.DataFrame(intervals, columns=['start', 'end', 'score'])
        print(pred_anomalies[['start','end']])
        pred_anomalies.to_csv(path+'pred_anomalies.csv')
        known_anomalies = casas_anomalies(y,x_index)
        anomalies = [pred_anomalies , known_anomalies]
        colors = ['red'] + ['green'] * (len(anomalies) - 1)
        plt.rcParams["figure.figsize"] = (50,10)
        fig = plt.figure()

        for i, anomaly in enumerate(anomalies):
          anomaly = list(anomaly[['start', 'end']].itertuples(index=False))

          for _, anom in enumerate(anomaly):
              t1 = convert_date_single(anom[0])
              t2 = convert_date_single(anom[1])
              plt.axvspan(t1, t2, color=colors[i], alpha=0.2)
              plt.plot(list(map(lambda x: convert_date_single(x), x_index[:len(y)] )), y)
        # plt.show()
        fig.savefig(path+'anomalies.png', dpi=fig.dpi)
        print('The plot with the anomalies is visible at {}'.format(path))

        tn, fp, fn, tp = contextual_confusion_matrix(known_anomalies, pred_anomalies, weighted=False )
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        print('precision: {}, recall: {}'.format(precision, recall))

        F1 = 2 * (precision * recall) / (precision + recall)
        gmean = np.sqrt(precision*recall)
        print('f1_score: {}, gmean: {}'.format(F1,gmean))


    else:
        df = pd.read_csv(read_path)

        x_index = index[0]
        true_index = x_index

        if not params.hyperbolic:
          error, true_index, true, pred = score_anomalies(true_signal, 
                                                          recons_signal, 
                                                          critic_score, 
                                                          x_index, 
                                                          rec_error_type=rec_error_type, 
                                                          comb=combination)
          torch.save(x_index, path+'true_index.pt')

        else:
          # true_data = torch.Tensor(recons_signal).reshape(-1,params.signal_shape)
          # pred_data = torch.Tensor(true_signal).reshape(-1,params.signal_shape)
          # sqdist = torch.sum((pred_data - true_data) ** 2, dim=1)
          # squnorm = torch.sum(pred_data ** 2, dim=-1)
          # sqvnorm = torch.sum(true_data ** 2, dim=-1)
          # x_temp = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 1e-7
          # error = torch.acosh(x_temp)

          # smoothing_window = math.trunc(true_signal.shape[0] * 0.01)
          # error = pd.Series(error).rolling(
          #         smoothing_window, center=True, min_periods=smoothing_window // 2).mean().values

          #using poincarÃ¨ distances between gt and preds as errors
          torch.save(x_index, path+'true_index.pt')
          # critic_smooth_window = math.trunc(recons_signal.shape[0] * 0.01)
          # critic_scores = _compute_critic_score(critic_score, critic_smooth_window)
          # error=error*critic_scores
          # recons_eucl = gmath.logmap0(torch.Tensor(eucl_recons), k=torch.tensor(-1.), dim=1).float()
          # real_eucl = gmath.logmap0(torch.Tensor(gt_signal), k=torch.tensor(-1.), dim=1).float()

          error, true_index, true, pred = score_anomalies(np.expand_dims(gt_signal,2), 
                                                          np.expand_dims(eucl_recons,2), 
                                                          critic_score, 
                                                          x_index, 
                                                          rec_error_type=rec_error_type, 
                                                          comb=combination)

        intervals = find_anomalies(error.reshape(-1), true_index, 
                                  window_size_portion=0.33, 
                                  window_step_size_portion=0.1, 
                                  fixed_threshold=True)
        
        try:
            anomalies = pd.DataFrame(intervals, columns=['start', 'end', 'score'])
            anomalies.to_csv(path+'anomalies.csv')

            plot(df, [anomalies, known_anomalies], signal, path)        
            print(contextual_confusion_matrix(known_anomalies, anomalies, data=df, weighted=False))
            out=list(contextual_confusion_matrix(known_anomalies, anomalies, data=df, weighted=False))

            tn, fp, fn, tp = contextual_confusion_matrix(known_anomalies, anomalies, weighted=False)
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)

            F1 = 2 * (precision * recall) / (precision + recall)

        except:
 
            # plot(df, [known_anomalies], signal, path)   
            out = [0,0,0,0]
            F1=0
        
        print('f1_score: {}'.format(F1))

        if params.save_result:
          file_place = './results/{}'.format(params.filename)

          if os.path.isfile(file_place):
            res = pd.read_csv(file_place) 
          else:
            res = pd.DataFrame(columns=['signal','fn','fp','tn','tp'])

          out=[signal]+out
          res.loc[len(res)] = out
          res.to_csv(file_place, index=False)


def test_lstm(test_loader, model, known_anomalies, path, signal = ''):
    df = pd.read_csv(path)
    # plot(df, known_anomalies,signal)
    recons_signal=[]
    true_signal=[]

    model = model.eval()
    for batch, (sample,index,y,y_index,x_index) in enumerate(test_loader):
        reconstructed_signal = model(sample.cuda().float())
        true_signal.append(sample.numpy())
        recons_signal.append(torch.squeeze(reconstructed_signal).cpu().detach().numpy())

    if len(recons_signal[-1].shape) == 1:
      recons_signal = np.append(np.concatenate(recons_signal[:-1]),recons_signal[-1].reshape(1,-1),axis=0)
    else:
      recons_signal = np.concatenate(recons_signal)

    true_signal = np.concatenate(true_signal)
    index = index[0] #[(index[0].shape[0]-y_index[0].shape[0]):]
    y=y[0]
    y_index = y_index[0]
    y_hat = unroll_ts(recons_signal)
    x_index = x_index[0]
    # pair-wise error calculation
    error = np.zeros(shape=y.shape)
    length = y.shape[0]
    for i in range(length):
        error[i] = abs(y_hat[i] - y[i])

    # visualize the error curve
    fig = plt.figure(figsize=(30, 3))
    plt.plot(error)
    fig.savefig('error.jpg')
    plt.show()

    errors, preds = reconstruction_errors(y.reshape(-1,1),y_hat.reshape(-1,1),rec_error_type="point")
    # # visualize the error curve
    # plot_error([[true, pred], error])
    # find anomalies
    print(errors)
    errors = stats.zscore(errors)
    errors = np.clip(errors, a_min=0, a_max=None) + 1
    print(errors)
    intervals = find_anomalies(error.reshape(-1), index.numpy(), 
                              window_size_portion=0.33, 
                              window_step_size_portion=0.1, 
                              fixed_threshold=True)
    # visualize the result
    anomalies = pd.DataFrame(intervals, columns=['start', 'end', 'score'])
    print(anomalies[['start','end']])
    print(known_anomalies)
    print(contextual_confusion_matrix(known_anomalies, anomalies, data=df, weighted=False))
    # find_scores(y_true, y_predict)
    plot(df, [anomalies, known_anomalies], signal)        


def casas_anomalies(y,x_index):
    # just loading casas anomalies in the correct format
    gt_anomal = []
    actual = None
    print(y.shape)
    y = y.reshape(y.shape[0]*y.shape[1], -1)[:x_index.shape[0]]
    for i in range(len(y.reshape(-1)==1)):
      if y[i]==1:
        if actual == None:
          actual = i
          start = x_index[i]
        else:
          actual = i
      else:
        if actual != None:
          gt_anomal.append((start, x_index[actual-1]))
          actual = None

    gt_anomal = pd.DataFrame.from_records(gt_anomal, columns =['start', 'end'])
    return gt_anomal

def convert_date_single(x):
    return datetime.fromtimestamp(x)

def _overlap(expected, observed):
    first = expected[0] - observed[1]
    second = expected[1] - observed[0]
    return first * second < 0

def _compute_critic_score(critics, smooth_window):
    """Compute an array of anomaly scores.
    Args:
        critics (ndarray):
            Critic values.
        smooth_window (int):
            Smooth window that will be applied to compute smooth errors.
    Returns:
        ndarray:
            Array of anomaly scores.
    """
    critics = np.asarray(critics)
    l_quantile = np.quantile(critics, 0.25)
    u_quantile = np.quantile(critics, 0.75)
    in_range = np.logical_and(critics >= l_quantile, critics <= u_quantile)
    critic_mean = np.mean(critics[in_range])
    critic_std = np.std(critics)

    z_scores = np.absolute((np.asarray(critics) - critic_mean) / critic_std) + 1
    z_scores = pd.Series(z_scores).rolling(
        smooth_window, center=True, min_periods=smooth_window // 2).mean().values

    return z_scores

def score_anomalies(y, y_hat, critic, index, score_window=10, critic_smooth_window=None,
                    error_smooth_window=None, smooth=True, rec_error_type="point", comb="mult",
                    lambda_rec=0.5,path=None):
    """Compute an array of anomaly scores.
    Anomaly scores are calculated using a combination of reconstruction error and critic score.
    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values. Each timestamp has multiple predictions.
        index (ndarray):
            time index for each y (start position of the window)
        critic (ndarray):
            Critic score. Each timestamp has multiple critic scores.
        score_window (int):
            Optional. Size of the window over which the scores are calculated.
            If not given, 10 is used.
        critic_smooth_window (int):
            Optional. Size of window over which smoothing is applied to critic.
            If not given, 200 is used.
        error_smooth_window (int):
            Optional. Size of window over which smoothing is applied to error.
            If not given, 200 is used.
        smooth (bool):
            Optional. Indicates whether errors should be smoothed.
            If not given, `True` is used.
        rec_error_type (str):
            Optional. The method to compute reconstruction error. Can be one of
            `["point", "area", "dtw"]`. If not given, 'point' is used.
        comb (str):
            Optional. How to combine critic and reconstruction error. Can be one
            of `["mult", "sum", "rec"]`. If not given, 'mult' is used.
        lambda_rec (float):
            Optional. Used if `comb="sum"` as a lambda weighted sum to combine
            scores. If not given, 0.5 is used.
    Returns:
        ndarray:
            Array of anomaly scores.
    """

    critic_smooth_window = critic_smooth_window or math.trunc(y.shape[0] * 0.01)
    error_smooth_window = error_smooth_window or math.trunc(y.shape[0] * 0.01)
    step_size = 1  # expected to be 1

    true_index = index  # no offset

    true = [item[0] for item in y.reshape((y.shape[0], -1))]

    for item in y[-1][1:]:
        true.extend(item)

    if (not path) or (path and not os.path.exists(path+'critic_scores.pickle')):
        critic_extended = list()
        for c in critic:
            critic_extended.extend(np.repeat(c, y_hat.shape[1]).tolist())
        critic_extended = np.asarray(critic_extended).reshape((-1, y_hat.shape[1]))
        
        critic_kde_max = []
        pred_length = y_hat.shape[1]
        num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] - 1)

        for i in range(num_errors):
            critic_intermediate = []

            for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
                critic_intermediate.append(critic_extended[i - j, j])

            if len(critic_intermediate) > 1:
                discr_intermediate = np.asarray(critic_intermediate)
                try:
              
                    critic_kde_max.append(discr_intermediate[np.argmax(
                        stats.gaussian_kde(discr_intermediate)(critic_intermediate))])
                except np.linalg.LinAlgError:
                    critic_kde_max.append(np.median(discr_intermediate))
            else:
                critic_kde_max.append(np.median(np.asarray(critic_intermediate)))

        # Compute critic scores
        critic_scores = _compute_critic_score(critic_kde_max, critic_smooth_window)
        
        
        if path:
          with open(path+'critic_scores.pickle', 'wb') as handle:
                pickle.dump(critic_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
          # with open(path+'rec_scores.pickle', 'wb') as handle:
          #       pickle.dump(rec_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
          with open(path+'critic_scores.pickle', 'rb') as handle:
            critic_scores = pickle.load(handle)
          # with open(path+'rec_scores.pickle', 'rb') as handle:
          #   rec_scores = pickle.load(handle)
          #   predictions = []
  
    if (not path) or (path and not os.path.exists(path+rec_error_type+'.pickle')):

        # Compute reconstruction scores
        rec_scores, predictions = reconstruction_errors(
            y, y_hat, step_size, score_window, error_smooth_window, smooth, rec_error_type)
        
        rec_scores = stats.zscore(rec_scores)
        rec_scores = np.clip(rec_scores, a_min=0, a_max=None) + 1

        if path:
          with open(path+rec_error_type+'.pickle', 'wb') as handle:
                pickle.dump(rec_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(path+rec_error_type+'.pickle', 'rb') as handle:
            rec_scores = pickle.load(handle)
            predictions=[]
    # Combine the two scores
    if comb == "mult":
        final_scores = np.multiply(critic_scores, rec_scores)

    elif comb == "sum":
        final_scores = (1 - lambda_rec) * (critic_scores - 1) + lambda_rec * (rec_scores - 1)

    elif comb == "rec":
        final_scores = rec_scores

    else:
        raise ValueError(
            'Unknown combination specified {}, use "mult", "sum", or "rec" instead.'.format(comb))

    true = [[t] for t in true]
    return final_scores, true_index, true, predictions

def _overlap_segment(expected, observed, start=None, end=None):
    tp, fp, fn = 0, 0, 0

    observed_copy = observed.copy()

    for expected_seq in expected:
        found = False
        for observed_seq in observed:
            if _overlap(expected_seq, observed_seq):
                if not found:
                    tp += 1
                    found = True
                if observed_seq in observed_copy:
                    observed_copy.remove(observed_seq)

        if not found:
            fn += 1

    fp += len(observed_copy)

    return None, fp, fn, tp


def _pad(lst):
    return [(part[0], part[1] + 1) for part in lst]


def contextual_confusion_matrix(expected, observed, data=None,
                                start=None, end=None, weighted=True):
    """Compute the confusion matrix between the ground truth and the detected anomalies.

    Args:
        expected (DataFrame or list of tuples):
            Ground truth passed as a ``pandas.DataFrame`` or list containing
            two columns: start and stop.
        observed (DataFrame or list of tuples):
            Detected anomalies passed as a ``pandas.DataFrame`` or list containing
            two columns: start and stop.
        data (DataFrame):
            Original data, passed as a ``pandas.DataFrame`` containing timestamp.
            Used to extract start and end.
        start (int):
            Minimum timestamp of the original data.
        end (int):
            Maximum timestamp of the original data.
        weighted (boolean):
            Flag to represent which algorithm to use.
            If true use weighted segment algorithm, else use overlap segment.

    Returns:
        tuple:
            number of true negative, false positive, false negative, true positive.
    """

    def _ws(x, y, z, w):
        return _weighted_segment(x, y, _contextual_partition, z, w)

    if weighted:
        function = _ws
    else:
        function = _overlap_segment

    if data is not None:
        start = data['timestamp'].min()
        end = data['timestamp'].max()

    if not isinstance(expected, list):
        expected = list(expected[['start', 'end']].itertuples(index=False))
    if not isinstance(observed, list):
        observed = list(observed[['start', 'end']].itertuples(index=False))

    expected = _pad(expected)
    observed = _pad(observed)

    return function(expected, observed, start, end)


def prune_false_positive(is_anomaly, anomaly_score, change_threshold):
    #The model might detect a high number of false positives.
    #In such a scenario, pruning of the false positive is suggested.
    #Method used is as described in the Section 5, part D Identifying Anomalous
    #Sequence, sub-part - Mitigating False positives
    #TODO code optimization
    seq_details = []
    delete_sequence = 0
    start_position = 0
    end_position = 0
    max_seq_element = anomaly_score[0]
    for i in range(1, len(is_anomaly)):
        if i+1 == len(is_anomaly):
            seq_details.append([start_position, i, max_seq_element, delete_sequence])
        elif is_anomaly[i] == 1 and is_anomaly[i+1] == 0:
            end_position = i
            seq_details.append([start_position, end_position, max_seq_element, delete_sequence])
        elif is_anomaly[i] == 1 and is_anomaly[i-1] == 0:
            start_position = i
            max_seq_element = anomaly_score[i]
        if is_anomaly[i] == 1 and is_anomaly[i-1] == 1 and anomaly_score[i] > max_seq_element:
            max_seq_element = anomaly_score[i]

    max_elements = list()
    for i in range(0, len(seq_details)):
        max_elements.append(seq_details[i][2])

    max_elements.sort(reverse=True)
    max_elements = np.array(max_elements)
    change_percent = abs(max_elements[1:] - max_elements[:-1]) / max_elements[1:]

    #Appending 0 for the 1 st element which is not change percent
    delete_seq = np.append(np.array([0]), change_percent < change_threshold)

    #Mapping max element and seq details
    for i, max_elt in enumerate(max_elements):
        for j in range(0, len(seq_details)):
            if seq_details[j][2] == max_elt:
                seq_details[j][3] = delete_seq[i]

    for seq in seq_details:
        if seq[3] == 1: #Delete sequence
            is_anomaly[seq[0]:seq[1]+1] = [0] * (seq[1] - seq[0] + 1)
 
    return is_anomaly

def detect_anomaly(anomaly_score):
    window_size = len(anomaly_score) // 3
    step_size = len(anomaly_score) // (3 * 10)

    is_anomaly = np.zeros(len(anomaly_score))

    for i in range(0, len(anomaly_score) - window_size, step_size):
        window_elts = anomaly_score[i:i+window_size]
        window_mean = np.mean(window_elts)
        window_std = np.std(window_mean)

        for j, elt in enumerate(window_elts):
            if (window_mean - 3 * window_std) < elt < (window_mean + 3 * window_std):
                is_anomaly[i + j] = 0
            else:
                is_anomaly[i + j] = 1

    return is_anomaly

def regression_errors(y, y_hat, smoothing_window=0.01, smooth=True):
    """Compute an array of absolute errors comparing predictions and expected output.

    If smooth is True, apply EWMA to the resulting array of errors.

    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values.
        smoothing_window (float):
            Optional. Size of the smoothing window, expressed as a proportion of the total
            length of y. If not given, 0.01 is used.
        smooth (bool):
            Optional. Indicates whether the returned errors should be smoothed with EWMA.
            If not given, `True` is used.

    Returns:
        ndarray:
            Array of errors.
    """
    errors = np.abs(y - y_hat)[:, 0]

    if not smooth:
        return errors

    smoothing_window = int(smoothing_window * len(y))

    return pd.Series(errors).ewm(span=smoothing_window).mean().values


def _point_wise_error(y, y_hat):
    """Compute point-wise error between predicted and expected values.

    The computed error is calculated as the difference between predicted
    and expected values with a rolling smoothing factor.

    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values.

    Returns:
        ndarray:
            An array of smoothed point-wise error.
    """
    return abs(y - y_hat)


def _area_error(y, y_hat, score_window=10):
    """Compute area error between predicted and expected values.

    The computed error is calculated as the area difference between predicted
    and expected values with a smoothing factor.

    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values.
        score_window (int):
            Optional. Size of the window over which the scores are calculated.
            If not given, 10 is used.

    Returns:
        ndarray:
            An array of area error.
    """
    smooth_y = pd.Series(y).rolling(
        score_window, center=True, min_periods=score_window // 2).apply(integrate.trapz)
    smooth_y_hat = pd.Series(y_hat).rolling(
        score_window, center=True, min_periods=score_window // 2).apply(integrate.trapz)

    errors = abs(smooth_y - smooth_y_hat)

    return errors


def _dtw_error(y, y_hat, score_window=10):
    """Compute dtw error between predicted and expected values.

    The computed error is calculated as the dynamic time warping distance
    between predicted and expected values with a smoothing factor.

    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values.
        score_window (int):
            Optional. Size of the window over which the scores are calculated.
            If not given, 10 is used.

    Returns:
        ndarray:
            An array of dtw error.
    """
    length_dtw = (score_window // 2) * 2 + 1
    half_length_dtw = length_dtw // 2
    
    # add padding
    y_pad = np.pad(y, (half_length_dtw, half_length_dtw),
                   'constant', constant_values=(0, 0))
    y_hat_pad = np.pad(y_hat, (half_length_dtw, half_length_dtw),
                       'constant', constant_values=(0, 0))

    i = 0
    similarity_dtw = list()
    while i < len(y) - length_dtw:
        true_data = y_pad[i:i + length_dtw]
        true_data = true_data.flatten()

        pred_data = y_hat_pad[i:i + length_dtw]
        pred_data = pred_data.flatten()
        dist = dtw(true_data, pred_data)
        similarity_dtw.append(dist)
        i += 1

    errors = ([0] * half_length_dtw + similarity_dtw +
              [0] * (len(y) - len(similarity_dtw) - half_length_dtw))

    return errors


def reconstruction_errors(y, y_hat, step_size=1, score_window=10, smoothing_window=0.01,
                          smooth=True, rec_error_type='point'):
    """Compute an array of reconstruction errors.

    Compute the discrepancies between the expected and the
    predicted values according to the reconstruction error type.

    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values. Each timestamp has multiple predictions.
        step_size (int):
            Optional. Indicating the number of steps between windows in the predicted values.
            If not given, 1 is used.
        score_window (int):
            Optional. Size of the window over which the scores are calculated.
            If not given, 10 is used.
        smoothing_window (float or int):
            Optional. Size of the smoothing window, when float it is expressed as a proportion
            of the total length of y. If not given, 0.01 is used.
        smooth (bool):
            Optional. Indicates whether the returned errors should be smoothed.
            If not given, `True` is used.
        rec_error_type (str):
            Optional. Reconstruction error types ``["point", "area", "dtw"]``.
            If not given, "point" is used.

    Returns:
        ndarray:
            Array of reconstruction errors.
    """
    if isinstance(smoothing_window, float):
        smoothing_window = min(math.trunc(len(y) * smoothing_window), 200)

    true = [item[0] for item in y.reshape((y.shape[0], -1))]
    for item in y[-1][1:]:
        true.extend(item)

    predictions = []
    predictions_vs = []

    pred_length = y_hat.shape[1]
    num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] - 1)

    for i in range(num_errors):
        intermediate = []
        for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
            intermediate.append(y_hat[i - j, j])
        if intermediate:
            predictions.append(np.median(np.asarray(intermediate)))

            predictions_vs.append([[
                np.min(np.asarray(intermediate)),
                np.percentile(np.asarray(intermediate), 25),
                np.percentile(np.asarray(intermediate), 50),
                np.percentile(np.asarray(intermediate), 75),
                np.max(np.asarray(intermediate))
            ]])

    true = np.asarray(true)
    predictions = np.asarray(predictions)
    predictions_vs = np.asarray(predictions_vs)

    # Compute reconstruction errors
    if rec_error_type.lower() == "point":
        errors = _point_wise_error(true, predictions)
        # errors = poincare_dist(true, predictions)

    elif rec_error_type.lower() == "area":
        errors = _area_error(true, predictions, score_window)

    elif rec_error_type.lower() == "dtw":
        errors = _dtw_error(true, predictions, score_window)


    # Apply smoothing
    if smooth:
        errors = pd.Series(errors).rolling(
            smoothing_window, center=True, min_periods=smoothing_window // 2).mean().values
    return errors, predictions_vs


def deltas(errors, epsilon, mean, std):
    """Compute mean and std deltas.

    delta_mean = mean(errors) - mean(all errors below epsilon)
    delta_std = std(errors) - std(all errors below epsilon)

    Args:
        errors (ndarray):
            Array of errors.
        epsilon (ndarray):
            Threshold value.
        mean (float):
            Mean of errors.
        std (float):
            Standard deviation of errors.

    Returns:
        float, float:
            * delta_mean.
            * delta_std.
    """
    below = errors[errors <= epsilon]
    if not len(below):
        return 0, 0

    return mean - below.mean(), std - below.std()


def count_above(errors, epsilon):
    """Count number of errors and continuous sequences above epsilon.

    Continuous sequences are counted by shifting and counting the number
    of positions where there was a change and the original value was true,
    which means that a sequence started at that position.

    Args:
        errors (ndarray):
            Array of errors.
        epsilon (ndarray):
            Threshold value.

    Returns:
        int, int:
            * Number of errors above epsilon.
            * Number of continuous sequences above epsilon.
    """
    above = errors > epsilon
    total_above = len(errors[above])

    above = pd.Series(above)
    shift = above.shift(1)
    change = above != shift

    total_consecutive = sum(above & change)

    return total_above, total_consecutive


def z_cost(z, errors, mean, std):
    """Compute how bad a z value is.

    The original formula is::

                 (delta_mean/mean) + (delta_std/std)
        ------------------------------------------------------
        number of errors above + (number of sequences above)^2

    which computes the "goodness" of `z`, meaning that the higher the value
    the better the `z`.

    In this case, we return this value inverted (we make it negative), to convert
    it into a cost function, as later on we will use scipy.fmin to minimize it.

    Args:
        z (ndarray):
            Value for which a cost score is calculated.
        errors (ndarray):
            Array of errors.
        mean (float):
            Mean of errors.
        std (float):
            Standard deviation of errors.

    Returns:
        float:
            Cost of z.
    """
    epsilon = mean + z * std

    delta_mean, delta_std = deltas(errors, epsilon, mean, std)
    above, consecutive = count_above(errors, epsilon)

    numerator = -(delta_mean / mean + delta_std / std)
    denominator = above + consecutive ** 2

    if denominator == 0:
        return np.inf

    return numerator / denominator


def _find_threshold(errors, z_range):
    """Find the ideal threshold.

    The ideal threshold is the one that minimizes the z_cost function. Scipy.fmin is used
    to find the minimum, using the values from z_range as starting points.

    Args:
        errors (ndarray):
            Array of errors.
        z_range (list):
            List of two values denoting the range out of which the start points for the
            scipy.fmin function are chosen.

    Returns:
        float:
            Calculated threshold value.
    """
    mean = errors.mean()
    std = errors.std()

    min_z, max_z = z_range
    best_z = min_z
    best_cost = np.inf
    for z in range(min_z, max_z):
        best = fmin(z_cost, z, args=(errors, mean, std), full_output=True, disp=False)
        z, cost = best[0:2]
        if cost < best_cost:
            best_z = z[0]

    return mean + best_z * std


def _fixed_threshold(errors, k=4):
    """Calculate the threshold.

    The fixed threshold is defined as k standard deviations away from the mean.

    Args:
        errors (ndarray):
            Array of errors.

    Returns:
        float:
            Calculated threshold value.
    """
    mean = errors.mean()
    std = errors.std()

    return mean + k * std


def _find_sequences(errors, epsilon, anomaly_padding):
    """Find sequences of values that are above epsilon.

    This is done following this steps:

        * create a boolean mask that indicates which values are above epsilon.
        * mark certain range of errors around True values with a True as well.
        * shift this mask by one place, filing the empty gap with a False.
        * compare the shifted mask with the original one to see if there are changes.
        * Consider a sequence start any point which was true and has changed.
        * Consider a sequence end any point which was false and has changed.

    Args:
        errors (ndarray):
            Array of errors.
        epsilon (float):
            Threshold value. All errors above epsilon are considered an anomaly.
        anomaly_padding (int):
            Number of errors before and after a found anomaly that are added to the
            anomalous sequence.

    Returns:
        ndarray, float:
            * Array containing start, end of each found anomalous sequence.
            * Maximum error value that was not considered an anomaly.
    """
    above = pd.Series(errors > epsilon)
    index_above = np.argwhere(above.values)

    for idx in index_above.flatten():
        above[max(0, idx - anomaly_padding):min(idx + anomaly_padding + 1, len(above))] = True

    shift = above.shift(1).fillna(False)
    change = above != shift

    if above.all():
        max_below = 0
    else:
        max_below = max(errors[~above])

    index = above.index
    starts = index[above & change].tolist()
    ends = (index[~above & change] - 1).tolist()

    if len(ends) == len(starts) - 1:
        ends.append(len(above) - 1)

    return np.array([starts, ends]).T, max_below


def _get_max_errors(errors, sequences, max_below):
    """Get the maximum error for each anomalous sequence.

    Also add a row with the max error which was not considered anomalous.

    Table containing a ``max_error`` column with the maximum error of each
    sequence and the columns ``start`` and ``stop`` with the corresponding start and stop
    indexes, sorted descendingly by the maximum error.

    Args:
        errors (ndarray):
            Array of errors.
        sequences (ndarray):
            Array containing start, end of anomalous sequences
        max_below (float):
            Maximum error value that was not considered an anomaly.

    Returns:
        pandas.DataFrame:
            DataFrame object containing columns ``start``, ``stop`` and ``max_error``.
    """
    max_errors = [{
        'max_error': max_below,
        'start': -1,
        'stop': -1
    }]

    for sequence in sequences:
        start, stop = sequence
        sequence_errors = errors[start: stop + 1]
        max_errors.append({
            'start': start,
            'stop': stop,
            'max_error': max(sequence_errors)
        })

    max_errors = pd.DataFrame(max_errors).sort_values('max_error', ascending=False)
    return max_errors.reset_index(drop=True)


def _prune_anomalies(max_errors, min_percent):
    """Prune anomalies to mitigate false positives.

    This is done by following these steps:

        * Shift the errors 1 negative step to compare each value with the next one.
        * Drop the last row, which we do not want to compare.
        * Calculate the percentage increase for each row.
        * Find rows which are below ``min_percent``.
        * Find the index of the latest of such rows.
        * Get the values of all the sequences above that index.

    Args:
        max_errors (pandas.DataFrame):
            DataFrame object containing columns ``start``, ``stop`` and ``max_error``.
        min_percent (float):
            Percentage of separation the anomalies need to meet between themselves and the
            highest non-anomalous error in the window sequence.

    Returns:
        ndarray:
            Array containing start, end, max_error of the pruned anomalies.
    """
    next_error = max_errors['max_error'].shift(-1).iloc[:-1]
    max_error = max_errors['max_error'].iloc[:-1]

    increase = (max_error - next_error) / max_error
    too_small = increase < min_percent

    if too_small.all():
        last_index = -1
    else:
        last_index = max_error[~too_small].index[-1]

    return max_errors[['start', 'stop', 'max_error']].iloc[0: last_index + 1].values


def _compute_scores(pruned_anomalies, errors, threshold, window_start):
    """Compute the score of the anomalies.

    Calculate the score of the anomalies proportional to the maximum error in the sequence
    and add window_start timestamp to make the index absolute.

    Args:
        pruned_anomalies (ndarray):
            Array of anomalies containing the start, end and max_error for all anomalies in
            the window.
        errors (ndarray):
            Array of errors.
        threshold (float):
            Threshold value.
        window_start (int):
            Index of the first error value in the window.

    Returns:
        list:
            List of anomalies containing start-index, end-index, score for each anomaly.
    """
    anomalies = list()
    denominator = errors.mean() + errors.std()

    for row in pruned_anomalies:
        max_error = row[2]
        score = (max_error - threshold) / denominator
        anomalies.append([row[0] + window_start, row[1] + window_start, score])

    return anomalies


def _merge_sequences(sequences):
    """Merge consecutive and overlapping sequences.

    We iterate over a list of start, end, score triples and merge together
    overlapping or consecutive sequences.
    The score of a merged sequence is the average of the single scores,
    weighted by the length of the corresponding sequences.

    Args:
        sequences (list):
            List of anomalies, containing start-index, end-index, score for each anomaly.

    Returns:
        ndarray:
            Array containing start-index, end-index, score for each anomaly after merging.
    """
    if len(sequences) == 0:
        return np.array([])

    sorted_sequences = sorted(sequences, key=lambda entry: entry[0])
    new_sequences = [sorted_sequences[0]]
    score = [sorted_sequences[0][2]]
    weights = [sorted_sequences[0][1] - sorted_sequences[0][0]]

    for sequence in sorted_sequences[1:]:
        prev_sequence = new_sequences[-1]

        if sequence[0] <= prev_sequence[1] + 1:
            score.append(sequence[2])
            weights.append(sequence[1] - sequence[0])
            weighted_average = np.average(score, weights=weights)
            new_sequences[-1] = (prev_sequence[0], max(prev_sequence[1], sequence[1]),
                                 weighted_average)
        else:
            score = [sequence[2]]
            weights = [sequence[1] - sequence[0]]
            new_sequences.append(sequence)

    return np.array(new_sequences)


def _find_window_sequences(window, z_range, anomaly_padding, min_percent, window_start,
                           fixed_threshold):
    """Find sequences of values that are anomalous.

    We first find the threshold for the window, then find all sequences above that threshold.
    After that, we get the max errors of the sequences and prune the anomalies. Lastly, the
    score of the anomalies is computed.

    Args:
        window (ndarray):
            Array of errors in the window that is analyzed.
        z_range (list):
            List of two values denoting the range out of which the start points for the
            dynamic find_threshold function are chosen.
        anomaly_padding (int):
            Number of errors before and after a found anomaly that are added to the anomalous
            sequence.
        min_percent (float):
            Percentage of separation the anomalies need to meet between themselves and the
            highest non-anomalous error in the window sequence.
        window_start (int):
            Index of the first error value in the window.
        fixed_threshold (bool):
            Indicates whether to use fixed threshold or dynamic threshold.

    Returns:
        ndarray:
            Array containing the start-index, end-index, score for each anomalous sequence
            that was found in the window.
    """
    if fixed_threshold:
        threshold = _fixed_threshold(window)

    else:
        threshold = _find_threshold(window, z_range)

    window_sequences, max_below = _find_sequences(window, threshold, anomaly_padding)
    max_errors = _get_max_errors(window, window_sequences, max_below)
    pruned_anomalies = _prune_anomalies(max_errors, min_percent)
    window_sequences = _compute_scores(pruned_anomalies, window, threshold, window_start)

    return window_sequences


def find_anomalies(errors, index, z_range=(0, 10), window_size=None, window_size_portion=None,
                   window_step_size=None, window_step_size_portion=None, min_percent=0.1,
                   anomaly_padding=50, lower_threshold=False, fixed_threshold=None):
    """Find sequences of error values that are anomalous.

    We first define the window of errors, that we want to analyze. We then find the anomalous
    sequences in that window and store the start/stop index pairs that correspond to each
    sequence, along with its score. Optionally, we can flip the error sequence around the mean
    and apply the same procedure, allowing us to find unusually low error sequences.
    We then move the window and repeat the procedure.
    Lastly, we combine overlapping or consecutive sequences.

    Args:
        errors (ndarray):
            Array of errors.
        index (ndarray):
            Array of indices of the errors.
        z_range (list):
            Optional. List of two values denoting the range out of which the start points for
            the scipy.fmin function are chosen. If not given, (0, 10) is used.
        window_size (int):
            Optional. Size of the window for which a threshold is calculated. If not given,
            `None` is used, which finds one threshold for the entire sequence of errors.
        window_size_portion (float):
            Optional. Specify the size of the window to be a portion of the sequence of errors.
            If not given, `None` is used, and window size is used as is.
        window_step_size (int):
            Optional. Number of steps the window is moved before another threshold is
            calculated for the new window.
        window_step_size_portion (float):
            Optional. Specify the number of steps to be a portion of the window size. If not given,
            `None` is used, and window step size is used as is.
        min_percent (float):
            Optional. Percentage of separation the anomalies need to meet between themselves and
            the highest non-anomalous error in the window sequence. It nof given, 0.1 is used.
        anomaly_padding (int):
            Optional. Number of errors before and after a found anomaly that are added to the
            anomalous sequence. If not given, 50 is used.
        lower_threshold (bool):
            Optional. Indicates whether to apply a lower threshold to find unusually low errors.
            If not given, `False` is used.
        fixed_threshold (bool):
            Optional. Indicates whether to use fixed threshold or dynamic threshold. If not
            given, `False` is used.

    Returns:
        ndarray:
            Array containing start-index, end-index, score for each anomalous sequence that
            was found.
    """
    window_size = window_size or len(errors)
    if window_size_portion:
        window_size = np.ceil(len(errors) * window_size_portion).astype('int')

    window_step_size = window_step_size or window_size
    if window_step_size_portion:
        window_step_size = np.ceil(window_size * window_step_size_portion).astype('int')

    window_start = 0
    window_end = 0
    sequences = list()
    
    while window_end < len(errors):
        window_end = window_start + window_size
        window = errors[window_start:window_end]
        window_sequences = _find_window_sequences(window, z_range, anomaly_padding, min_percent,
                                                  window_start, fixed_threshold)
        sequences.extend(window_sequences)

        if lower_threshold:
            # Flip errors sequence around mean
            mean = window.mean()
            inverted_window = mean - (window - mean)
            inverted_window_sequences = _find_window_sequences(inverted_window, z_range,
                                                               anomaly_padding, min_percent,
                                                               window_start, fixed_threshold)
            sequences.extend(inverted_window_sequences)

        window_start = window_start + window_step_size

    sequences = _merge_sequences(sequences)

    anomalies = list()

    # for start, stop, score in sequences:

    #     if start < index.shape[0]:
    #       if stop < index.shape[0]:pass
    #       else: stop = index.shape[0]-1
    #       anomalies.append([index[int(start)], index[int(stop)], score])
    for start, stop, score in sequences:
        anomalies.append([index[int(start)], index[int(stop)], score])

    return np.asarray(anomalies)


def find_scores(y_true, y_predict):
    tp = tn = fp = fn = 0

    for i in range(0, len(y_true)):
        if y_true[i] == 1 and y_predict[i] == 1:
            tp += 1
        elif y_true[i] == 1 and y_predict[i] == 0:
            fn += 1
        elif y_true[i] == 0 and y_predict[i] == 0:
            tn += 1
        elif y_true[i] == 0 and y_predict[i] == 1:
            fp += 1

    print ('Accuracy {:.2f}'.format((tp + tn)/(len(y_true))))
    try:
      precision = tp / (tp + fp)
      recall = tp / (tp + fn)
      print ('Precision {:.2f}'.format(precision))
      print ('Recall {:.2f}'.format(recall))
      print ('F1 Score {:.2f}'.format(2 * precision * recall / (precision + recall)))
    except:
      print ('Precision {:.2f}'.format(0))
      print ('Recall {:.2f}'.format(0))
      print ('F1 Score {:.2f}'.format(0))


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

      else:
        train_data = od.load_signal(params.signal +'-train')
        test_data = od.load_signal(params.signal +'-test')

        train_dataset = SignalDataset(path='./data/{}-train.csv'.format(params.signal),interval=21600)
        test_dataset = SignalDataset(path='./data/{}-test.csv'.format(params.signal),interval=21600,test=True)
        demo=False

    batch_size = params.batch_size
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle=False,num_workers=6)

    lr = params.lr

    
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
      signal_shape = params.signal_shape
      if params.hyperbolic: PATH = './models_{}_{}_{}/'.format(params.dataset,str(params.epochs),str(params.lr))+ params.signal
      else: PATH = './models_eucl_{}_{}_{}/'.format(params.dataset,str(params.epochs),str(params.lr))+ params.signal

      encoder = torch.load(PATH+'/encoder.pt').cuda()
      decoder = torch.load(PATH+'/decoder.pt').cuda()
      critic_x = torch.load(PATH+'/critic_x.pt').cuda()
      critic_z = torch.load(PATH+'/critic_z.pt').cuda()
      
      encoder.eval()
      decoder.eval()
      critic_x.eval()
      critic_z.eval()

      if params.dataset not in ['CASAS', 'new_CASAS']:
        if params.signal in signal_list:
          known_anomalies = pd.DataFrame()
        else:
          known_anomalies = od.load_anomalies(params.signal)
      else: known_anomalies=[]

      if demo:
        test_tadgan(test_loader, encoder, decoder, critic_x, critic_z, known_anomalies, read_path=path.format(params.signal), signal = params.signal, hyperbolic = params.hyperbolic, path=PATH, params=params)
      else:
        test_tadgan(test_loader, encoder, decoder, critic_x, critic_z, known_anomalies, read_path='./data/{}-test.csv'.format(params.signal),signal = params.signal, hyperbolic = params.hyperbolic, path=PATH, signal_shape=signal_shape, params=params)
