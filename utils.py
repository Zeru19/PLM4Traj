import math
import os
import shutil
from itertools import count, islice, tee
from os.path import exists
from datetime import datetime
from collections import namedtuple
from copy import deepcopy

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, recall_score, accuracy_score, roc_auc_score
from einops import rearrange, repeat


def idDocker():
    return os.path.exists('/.dockerenv')


def create_if_noexists(*paths):
    """ Create directories if they don't exist already. """
    for path in paths:
        if not exists(path):
            os.makedirs(path)


def remove_if_exists(path):
    """ Remove a file if it exists. """
    if exists(path):
        os.remove(path)


def next_batch(data, batch_size):
    """ Yield the next batch of given data. """
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        if end_index - start_index > 1:
            yield data[start_index:end_index]


def next_batch_recycle(data, batch_size):
    """ Yield the next batch of given data, and it can recycle"""
    assert batch_size > 0
    data_length = len(data)
    assert batch_size <= data_length, 'Batch size is large than total data length. ' \
                                      'Please check your data or change batch size.'
    batch_index = 0
    while True:
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        # if end_index - start_index > 1:
        if end_index - start_index == batch_size:
            yield data[start_index:end_index]
            batch_index += 1
        else:  # recycle(no use the last incomplete batch)
            batch_index = 0


def clean_dirs(*dirs):
    """ Remove the given directories, including all contained files and sub-directories. """
    for dir in dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)


def mean_absolute_percentage_error(y_true, y_pred):
    """ Calculcates the MAPE metric. """
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    return mape


def cal_regression_metric(label, pres):
    """ Calculcate all common regression metrics. """
    rmse = math.sqrt(mean_squared_error(label, pres))
    mae = mean_absolute_error(label, pres)
    mape = mean_absolute_percentage_error(label, pres)

    s = pd.Series([rmse, mae, mape], index=['rmse', 'mae', 'mape'])
    return s


def top_n_accuracy(truths, preds, n):
    """ Calculcate Acc@N metric. """
    best_n = np.argsort(preds, axis=1)[:, -n:]
    successes = 0
    for i, truth in enumerate(truths):
        if truth in best_n[i, :]:
            successes += 1
    return float(successes) / truths.shape[0]


def cal_classification_metric(labels, pres):
    """
    Calculates all common classification metrics.

    :param labels: classification label, with shape (N).
    :param pres: predicted classification distribution, with shape (N, num_class).
    """
    mean_rank = cal_mean_rank(pres, labels)
    pres_index = pres.argmax(-1)  # (N)
    macro_f1 = f1_score(labels, pres_index, average='macro', zero_division=0)
    macro_recall = recall_score(labels, pres_index, average='macro', zero_division=0)
    acc = accuracy_score(labels, pres_index)
    n_list = [5, 10]
    top_n_acc = [top_n_accuracy(labels, pres, n) for n in n_list]

    s = pd.Series([mean_rank, macro_f1, macro_recall, acc] + top_n_acc,
                  index=['mean_rank', 'macro_f1', 'macro_rec'] +
                  [f'acc@{n}' for n in [1] + n_list])
    return s


def intersection(lst1, lst2):
    """ Calculates the intersection of two sets, or lists. """
    lst3 = list(set(lst1) & set(lst2))
    return lst3


def get_datetime_key():
    """ Get a string key based on current datetime. """
    return 't' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def cal_model_size(model):
    """ Calculate the total size (in megabytes) of a torch module. """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def geo_distance(lng1, lat1, lng2, lat2):
    """ Calculcate the geographical distance between two points (or one target point and an array of points). """
    lng1, lat1, lng2, lat2 = map(np.radians, [lng1, lat1, lng2, lat2])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    distance = 2 * np.arcsin(np.sqrt(a)) * 6371 * 1000
    return distance


def normalization_torch(data):
    _range = torch.max(data) - torch.min(data)
    return (data - torch.min(data)) / _range


def calc_azimuth(lat1, lon1, lat2, lon2):
    import math

    lat1_rad = lat1 * math.pi / 180
    lon1_rad = lon1 * math.pi / 180
    lat2_rad = lat2 * math.pi / 180
    lon2_rad = lon2 * math.pi / 180

    y = math.sin(lon2_rad - lon1_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon2_rad - lon1_rad)

    brng = math.atan2(y, x) * 180 / math.pi

    return float((brng + 360.0) % 360.0)


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def cal_mean_rank(scores, target_indices):
    """
    Calculate the Mean Rank metric.

    :param scores: A 2D NumPy array where each row contains the predicted scores for each label.
    :param target_indices: A 1D NumPy array containing the index of the target item in each prediction.
    :return: The value of Mean Rank.
    """
    # Get the ranks of each score in descending order
    ranks = scores.argsort(axis=1)[:, ::-1].argsort(axis=1) + 1

    # Extract the ranks of the target indices
    target_indices = target_indices.astype(int)
    target_ranks = ranks[np.arange(len(target_indices)), target_indices]

    # Calculate the mean of these ranks
    mean_rank_value = np.mean(target_ranks)
    return mean_rank_value
