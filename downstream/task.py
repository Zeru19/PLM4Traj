import time

import nni
import numpy as np
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange

from downstream.trainer import *


class Destination(Trainer):
    """
    A helper class for destination prediction.
    Feeds the encoders with truncated trajectories,
    then regard the destinations of trajectories (last point) as prediction target.
    """

    def __init__(self, pre_length, **kwargs):
        super().__init__(task_name='destination', metric_type='classification', **kwargs)
        self.pre_length = pre_length
        self.loss_func = F.cross_entropy

    def forward_encoders(self, *x, **kwargs):
        if kwargs.get('lang', 'zh') == 'zh':
            suffix_prompt="目的地所在路段为"
        else:
            suffix_prompt="The destination is"

        if len(x) < 2:
            return super().forward_encoders(*x, suffix_prompt=suffix_prompt, d_mask=True, **kwargs)

        trip, valid_len = x[:2]
        return super().forward_encoders(trip, valid_len-self.pre_length, *x[2:], suffix_prompt=suffix_prompt, d_mask=True,
                                        **kwargs)

    def parse_label(self, label_meta):
        return label_meta.long().detach()


class TTE(Trainer):
    """
    A helper class for travel time estimation evaluation.
    The prediction targets is the time span (in minutes) of trajectories.
    """

    def __init__(self, **kwargs):
        super().__init__(task_name=f'tte', metric_type='regression', **kwargs)
        self.loss_func = F.mse_loss

    def forward_encoders(self, *x, **kwargs):
        if kwargs.get('lang', 'zh') == 'zh':
            suffix_prompt="旅行时间为"
        else:
            suffix_prompt="The total travel time is"

        if len(x) < 2:
            return super().forward_encoders(*x, suffix_prompt=suffix_prompt)

        trip, valid_len = x[:2]
        return super().forward_encoders(trip, valid_len, *x[2:], suffix_prompt=suffix_prompt)

    def parse_label(self, label_meta):
        return label_meta.float()
