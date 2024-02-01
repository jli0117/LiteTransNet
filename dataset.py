from typing import Optional

import numpy as np
from torch.utils.data import Dataset
import torch

from src.utils.utils import normalizer, slicing_window, loading_data


class LSDataset(Dataset):
    def __init__(self,
                 name: str,
                 **kwargs):
    
        super().__init__(**kwargs)

        self.name = name

        self._load_csv()

    def _load_csv(self):

        # load data
        dataset = loading_data(self.name)
        data = dataset.values

        n_months = 12

        # normalisation
        normed_data, min_val, max_val = normalizer(data)

        # split into 3d array, the label is the next row
        features, labels = slicing_window(normed_data, n_months)

        # Store feature and label
        self._x = features
        self._y = labels

        # Convert to float32 (output)
        self._x = torch.Tensor(self._x)
        self._y = torch.Tensor(self._y)

        self._maxvalue = max_val
        self._minvalue = min_val

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self._x[idx], self._y[idx])

    def __len__(self):
        return self._x.shape[0]
    
