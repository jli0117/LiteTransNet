import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_loss(net: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 loss_function: torch.nn.Module,
                 device: torch.device = 'cpu') -> torch.Tensor:
    running_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            netout = net(x.to(device)).cpu()
            running_loss += loss_function(y, netout)

    return running_loss / len(dataloader)


def normalizer(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # norm_data = numerator / (denominator + 1e-7)
    norm_data = numerator / denominator
    return norm_data, np.min(data, 0), np.max(data, 0)


def scaler(data):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    data_scaled = scaler.fit_transform(data)
    min_value = scaler.data_min_
    max_value = scaler.data_max_
    return data_scaled, min_value, max_value
    

def rescaler(data, min_value, max_value):
    inv_y = (data - (-1)) * (max_value - min_value) / (1 - (-1)) + min_value
    return inv_y

def renormlizer(data, max_val, min_val):
    data = data * (max_val - min_val)
    data = data + min_val
    return data


def slicing_window(data, n_in):
    list_of_features = []
    list_of_labels = []

    for i in range(len(data)-n_in+1):
        arr_features = data[i:(i+n_in), :-1]
        arr_label = data[i:(i+n_in), -1]
        list_of_features.append(arr_features)
        list_of_labels.append(arr_label.reshape(-1, 1))

    features = np.array(list_of_features)
    labels = np.array(list_of_labels) 
    return features, labels

