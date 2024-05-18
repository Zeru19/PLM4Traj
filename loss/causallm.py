import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from einops import rearrange, repeat, reduce

from .base import Loss


class TripLoss(Loss):
    """ Predict the trip points by od pois. """

    def __init__(self, latent_size, out_dis, out_con_feas, 
                 dis_weight=0.5, con_weight=0.5):
        super().__init__(name=f'trip-dis-{dis_weight}-' + ','.join(map(str, out_dis['feats'])) + \
                              (f'-con-{con_weight}-' + ','.join(map(str, out_con_feas)) if out_con_feas else '')
                         )
        
        self.dis_feats = out_dis['feats']
        self.dis_num_classes = out_dis['num_embeds']
        self.con_feats = out_con_feas
        self.dis_weight = dis_weight
        self.con_weight = con_weight

        self.dis_pres = nn.ModuleList([nn.Sequential(nn.Linear(latent_size, latent_size * 4, bias=False),
                                                     nn.LayerNorm(latent_size * 4),
                                                     nn.LeakyReLU(inplace=True),
                                                     nn.Linear(latent_size * 4, num_class),
                                                     nn.Softplus())
                                       for num_class in self.dis_num_classes])
        
        self.con_pre = nn.Sequential(nn.Linear(latent_size, latent_size // 4, bias=False),
                                     nn.LayerNorm(latent_size // 4),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(latent_size // 4, len(out_con_feas)))
        
    def forward(self, latent, trip, valid_len, **kwargs):
        # Calculate recovery loss.
        loss = 0.0
        for i, dis_pre in enumerate(self.dis_pres):
            pred, label = dis_pre(latent), self._flat(True, trip[..., self.dis_feats[i]].long(), length=valid_len)[0]
            loss += self.dis_weight * F.cross_entropy(pred, label, reduction='mean')
        if len(self.con_feats) > 0:
            pred, label = self.con_pre(latent), self._flat(True, trip[..., self.con_feats], length=valid_len)[0]
            loss += self.con_weight * F.mse_loss(pred, label, reduction='mean')
        return loss
    
    @staticmethod
    def _flat(flat_valid, *xs, length):
        if flat_valid:
            return [pack_padded_sequence(x, length.long().cpu(), batch_first=True, enforce_sorted=False).data
                    for x in xs]
        else:
            return [x.reshape(-1, x.size(-1)) if len(x.shape) > 2 else x.reshape(-1) for x in xs]
        

class POILoss(Loss):
    def __init__(self, latent_size):
        super().__init__(name='poi')

        self.loss_fct = nn.CrossEntropyLoss()
        
    def forward(self, logits, labels):
        # Calculate recovery loss.
        return self.loss_fct(logits.view(-1, logits.size(-1)), labels.long().view(-1))


class NoneLoss(Loss):
    def __init__(self):
        super().__init__(name='none')

    def forward(self, *args, **kwargs):
        return 0.0


class TripCausalLoss(Loss):
    def __init__(self, latent_size, out_dis, out_con_feas, 
                 dis_weight=0.5, con_weight=0.5, shift_labels=False, add_poi=True):

        self.add_poi = add_poi

        trip_loss = TripLoss(latent_size, out_dis, out_con_feas, dis_weight, con_weight)
        if add_poi:
            poi_loss = POILoss(latent_size)
        else:
            poi_loss = NoneLoss()

        super().__init__(name=f'trip_causual-{trip_loss.name}-{poi_loss.name}')
        self.trip_loss = trip_loss
        self.poi_loss = poi_loss
        self.shift_labels = shift_labels

    def forward(self, models, trip, valid_len, o_pois, d_pois, start_weekday, start_hour, **kwargs):
        encoder, = models
        # Feed encode metas to the encoder.
        trip_latent, o_logits, d_logits, o_labels, d_labels = \
            encoder.forward_flip(trip, valid_len, o_pois, d_pois, start_weekday, start_hour, shift_labels=self.shift_labels)
        
        if self.shift_labels:
            trip = trip[:, 1:]
            valid_len = valid_len - 1
        trip_loss = self.trip_loss(trip_latent, trip, valid_len)
        poi_loss = self.poi_loss(o_logits, o_labels) + self.poi_loss(d_logits, d_labels)

        return trip_loss + poi_loss
    