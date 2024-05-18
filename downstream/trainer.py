import sys
import math
from abc import abstractmethod

import nni
import copy
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm, trange
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from utils import create_if_noexists, cal_classification_metric, cal_regression_metric, mean_absolute_percentage_error
from data import SET_NAMES
from pretrain.trainer import Trainer as PreTrainer


class Trainer:
    """
    Base class of the downstream trainer.
    Implements most of the functions shared by various downstream tasks.
    """
    def __init__(self, task_name, base_name, metric_type, train_data, eval_data, models,
                 predictor, num_epoch, lr, device, log_name_key, cache_dir,
                 es_epoch=-1, finetune=False, save_prediction=False, **kwargs):
        self.task_name = task_name
        self.metric_type = metric_type
        self.cache_dir = cache_dir

        self.train_dataloader = train_data
        self.eval_dataloader = eval_data
        # All models feed into the downstream trainer will be used for calculating the embedding vectors.
        # The embedding vectors will be concatenated along the feature dimension.
        self.models = [model.to(device) for model in models]
        # The predictor is fed with the embedding vectors, and output the prediction required by the downstream task.
        self.predictor = predictor.to(device)

        self.use_nni = bool(kwargs.get('use_nni', False))
        self.disable_tqdm = True if self.use_nni else False
        self.batch_size = train_data.batch_size
        self.es_epoch = es_epoch
        self.num_epoch = num_epoch
        self.lr = lr
        self.device = device
        self.log_name_key = log_name_key

        self.finetune = finetune
        self.save_prediction = save_prediction

        model_name = '_'.join([f'{model.name}-ds' for model in models])
        self.base_key = f'{task_name}/{base_name}/{model_name}_ft{int(finetune)}'
        self.model_cache_dir = f'{cache_dir}/model_cache/{self.base_key}'
        self.model_save_dir = f'{cache_dir}/model_save/{self.base_key}'
        self.log_save_dir = f'{cache_dir}/log/{self.base_key}'
        self.pred_save_dir = f'{cache_dir}/pred/{self.base_key}'

        self.optimizer = torch.optim.Adam(PreTrainer.gather_all_param(*self.models, self.predictor), lr=lr)

        self.downstream_token = nn.Parameter(torch.zeros(self.models[0].emb_size).float(), requires_grad=True).to(device)

    def train(self):
        num_noimprove_epochs = 0
        best_metric = 0.0
        train_logs = []
        desc_text = 'Downstream training, val metric %.4f'
        with trange(self.num_epoch, desc=desc_text % 0.0, disable=self.disable_tqdm) as tbar:
            best_model_wts = self.get_model_state_dicts()
            for epoch_i in tbar:
                train_loss = self.train_epoch()
                val_metric, es_metric = self.eval(1, full_metric=False)

                # Report intermediate result if using NNI (Neural Network Intelligence)
                if self.use_nni:
                    nni.report_intermediate_result(es_metric)

                train_logs.append([epoch_i, val_metric, train_loss])
                tbar.set_description(desc_text % val_metric)

                # Check and update the best model
                if es_metric > best_metric:
                    best_metric = es_metric
                    num_noimprove_epochs = 0
                    best_model_wts = self.get_model_state_dicts()  # Update the best model weights
                else:
                    num_noimprove_epochs += 1

                # Early stopping condition
                if self.es_epoch > -1 and num_noimprove_epochs >= self.es_epoch:
                    # Report final result if using NNI
                    if self.use_nni:
                        nni.report_final_result(best_metric)

                    self.set_model_state_dicts(best_model_wts)  # Load the best model weights
                    tbar.set_description('Early stopped')
                    break

        self.save_models()
        # Save training logs to a HDF5 file.
        create_if_noexists(self.log_save_dir)
        train_logs = pd.DataFrame(train_logs, columns=['epoch', 'val_metric', 'loss'])
        train_logs.to_hdf(f'{self.log_save_dir}/{self.log_name_key}.h5', key='downstream_train_log')

        return self.models, self.predictor

    def train_epoch(self):
        self.train_state()

        loss_log = []
        for batch_meta in tqdm(self.train_dataloader,
                               desc=f'-->Traverse batches', total=len(self.train_dataloader), leave=False,
                               disable=self.disable_tqdm,
                               file=sys.stdout):
            batch_meta = [e.to(self.device) if isinstance(e, torch.Tensor) else e for e in batch_meta]
            enc_meta, label = batch_meta[:-1], batch_meta[-1]
            encodes = self.forward_encoders(*enc_meta)
            pre = self.predictor(encodes).squeeze(-1)

            label = self.parse_label(label)
            loss = self.loss_func(pre, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_log.append(loss.item())
        return float(np.mean(loss_log))

    def eval(self, set_index, full_metric=True):
        set_name = SET_NAMES[set_index][1]
        self.eval_state()

        pres, labels = [], []
        for batch_meta in tqdm(self.eval_dataloader,
                               desc=f'Evaluating on {set_name} set',
                               total=len(self.eval_dataloader), leave=False,
                               disable=self.disable_tqdm):
            batch_meta = [e.to(self.device) if isinstance(e, torch.Tensor) else e for e in batch_meta]
            enc_meta, label = batch_meta[:-1], batch_meta[-1]
            label = self.parse_label(label)
            encodes = self.forward_encoders(*enc_meta)
            pre = self.predictor(encodes).squeeze(-1)

            pres.append(pre.detach().cpu().numpy())
            labels.append(label.cpu().numpy())
        pres, labels = np.concatenate(pres, 0), np.concatenate(labels, 0)

        if full_metric:
            self.metric_and_save(labels, pres, set_name)
        else:
            if self.metric_type == 'regression':
                mape = mean_absolute_percentage_error(labels, pres)
                return mape, 1 / (mape + 1e-6)
            elif self.metric_type == 'classification':
                acc = accuracy_score(labels, pres.argmax(-1))
                return acc, acc

    @abstractmethod
    def parse_label(self, label_meta):
        pass

    def loss_func(self, pre, label):
        pass

    def forward_encoders(self, *x, **kwargs):
        """ Feed the input to all encoders and concatenate the embedding vectors.  """
        encodes = [encoder(*x, token=self.downstream_token, **kwargs) for encoder in self.models]
        if not self.finetune:
            encodes = [encode.detach() for encode in encodes]
        encodes = torch.cat(encodes, -1)
        return encodes  # (B, num_encoders * E)

    def train_state(self):
        """ Turn all models and the predictor into training mode.  """
        for encoder in self.models:
            encoder.train()
        self.predictor.train()

    def eval_state(self):
        """ Turn all models and the predictor into evaluation mode.  """
        for encoder in self.models:
            encoder.eval()
        self.predictor.eval()

    def save_models(self, epoch=None):
        """ Save the encoder model and the predictor model. """
        for model in (*self.models, self.predictor):
            if epoch is not None:
                create_if_noexists(self.model_cache_dir)
                save_path = f'{self.model_cache_dir}/{model.name}_epoch{epoch}.model'
            else:
                create_if_noexists(self.model_save_dir)
                save_path = f'{self.model_save_dir}/{model.name}.model'
                print('Saved model', model.name)
            torch.save(model.state_dict(), save_path)

    def load_model(self, model, epoch=None):
        """ Load one of the encoder. """
        if epoch is not None:
            save_path = f'{self.model_cache_dir}/{model.name}_epoch{epoch}.model'
        else:
            save_path = f'{self.model_save_dir}/{model.name}.model'
            print('Load model', model.name)
        model.load_state_dict(torch.load(save_path, map_location=self.device))
        return model

    def load_models(self, epoch=None):
        """ Load all encoders. """
        for i, encoder in enumerate(self.models):
            self.models[i] = self.load_model(encoder, epoch)
        self.predictor = self.load_model(self.predictor, epoch)

    def get_model_state_dicts(self):
        """ Get the state dicts of all encoders and predictor. """
        state_dict_list = []
        for model in (*self.models, self.predictor):
            state_dict_list.append(model.state_dict())
        return state_dict_list

    def set_model_state_dicts(self, state_dict_list):
        """ Set the state dicts of all encoders and predictor. """
        for i, model in enumerate((*self.models, self.predictor)):
            model.load_state_dict(state_dict_list[i])

    def metric_and_save(self, labels, pres, save_name):
        """ Calculate the evaluation metric, then save the metric and the prediction result. """
        if self.metric_type == 'classification':
            metric = cal_classification_metric(labels, pres)
        elif self.metric_type == 'regression':
            metric = cal_regression_metric(labels, pres)
        else:
            raise NotImplementedError(f'No type "{type}".')
        print(metric)

        create_if_noexists(self.log_save_dir)
        metric.to_hdf(f'{self.log_save_dir}/{save_name}_{self.log_name_key}.h5',
                      key='metric', format='table')

        if self.save_prediction:
            create_if_noexists(self.pred_save_dir)
            np.savez(f'{self.pred_save_dir}/{save_name}_{self.log_name_key}.npz',
                     labels=labels, pres=pres)
