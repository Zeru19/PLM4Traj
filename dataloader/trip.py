import math
import random
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from einops import repeat, rearrange


def is_str_array(var):
    return isinstance(var, np.ndarray) and var.dtype.type in [np.unicode_, np.string_]


class TripODPOIWithHour(Dataset):
    def __init__(self, trip, o_pois=None, d_pois=None, label=None, prop=1.0):
        super().__init__()
        self.name = 'TripODPOIWithHour'

        self.prop = prop
        if prop < 1:
            self.name += f'_prop{prop}'

        length = len(trip)
        self.trip = trip[:int(length * prop)]
        self.o_pois = o_pois[:int(length * prop)] if o_pois is not None else None
        self.d_pois = d_pois[:int(length * prop)] if d_pois is not None else None

        self.label = label[:int(length * prop)] if label is not None else None

    def __len__(self):
        return len(self.trip)
    
    def __getitem__(self, index):
        trip = self.trip[index]
        time_array = pd.to_datetime(trip[:, 7], unit='s')
        weekday = int(time_array.weekday.to_numpy()[0])
        hour = int(time_array.hour.to_numpy()[0])
        length = len(trip)
        
        # seconds to hours
        trip[:, 7] = time_array.hour.to_numpy()

        # If the embeddings are not provided, return None.
        if self.o_pois is not None:
            o_pois = self.o_pois[index][0]
        else:
            o_pois = None

        if self.d_pois is not None:
            d_pois = self.d_pois[index][0]
        else:
            d_pois = None

        if self.label is not None:
            label = self.label[index]
            return trip, length, o_pois, d_pois, weekday, hour, label
        else:
            return trip, length, o_pois, d_pois, weekday, hour
        
    @staticmethod
    def collate_fn(batch):
        if len(batch[0]) == 7:
            trip, length, o_embedding, d_embedding, weekday, hour, label = zip(*batch)
            label = torch.tensor(label).float()
        else:
            trip, length, o_embedding, d_embedding, weekday, hour = zip(*batch)

        trip = TripODPOIWithHour.pad_trip(trip, length)
        valid_lens = torch.tensor(length).long()

        if o_embedding is not None:
            o_embedding = np.stack(o_embedding)
            d_embedding = np.stack(d_embedding)
            o_embedding = o_embedding.tolist() if is_str_array(o_embedding) else torch.tensor(o_embedding).float()
            d_embedding = d_embedding.tolist() if is_str_array(d_embedding) else torch.tensor(d_embedding).float()
        else:
            o_embedding = None
            d_embedding = None

        weekday = torch.tensor(weekday).long()
        hour = torch.tensor(hour).long()

        if len(batch[0]) == 7:
            return trip, valid_lens, o_embedding, d_embedding, weekday, hour, label
        else:
            return trip, valid_lens, o_embedding, d_embedding, weekday, hour
        
    @staticmethod
    def pad_trip(trips, valid_lens):
        max_len = max(valid_lens)
        padded_trips = []
        for trip in trips:
            # trip = trip + [trip[-1]] * (max_len - len(trip))
            trip = np.concatenate([trip, repeat(trip[-1], 'd -> l d', l=max_len-len(trip))])
            padded_trips.append(trip)
        padded_trips = np.stack(padded_trips)
        padded_trips = torch.from_numpy(padded_trips).float()

        return padded_trips
    