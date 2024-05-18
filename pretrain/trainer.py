import math
from time import time
from abc import abstractmethod

import nni
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from tqdm import tqdm, trange

from utils import create_if_noexists, cal_model_size
from data import SET_NAMES


class Trainer:
    """
    Base class of the pre-training helper class.
    Implements most of the functions shared by all types of pre-trainers.
    """

    def __init__(self, dataloader, meta_name, models, trainer_name,
                 loss_func, num_epoch, lr, device,
                 log_name_key, cache_dir, cache_epoches=False,
                 suffix='', **kwargs):
        """
        :param dataloader: torch dataloader object.
        :param meta_name: name of the meta types.
        :param models: list of models. Depending on the type of pretext task, they can be encoders or decoders.
        :param trainer_name: name of the pre-trainer.
        :param loss_func: loss function module defined by specific pretext task.
        :param log_name_key: the name key for saving training logs. All saved log will use this key as their file name.
        :param cache_epoches: whether to save all models after every training epoch.
        """
        self.dataloader = dataloader
        self.cache_dir = cache_dir

        # The list of models may have different usage in different types of trainers.
        self.models = [model.to(device) for model in models]
        self.trainer_name = trainer_name

        self.use_nni = bool(kwargs.get('use_nni', False))
        self.disable_tqdm = True if self.use_nni else False
        self.num_epoch = num_epoch
        self.lr = lr
        self.device = device
        self.cache_epoches = cache_epoches
        self.loss_func = loss_func.to(device)
        loss_name = loss_func.name

        model_name = '_'.join([model.name for model in models])
        self.BASE_KEY = f'{trainer_name}_b{dataloader.batch_size}-lr{lr}{suffix}/{loss_name}/{meta_name}/{model_name}'
        self.model_cache_dir = f'{cache_dir}/model_cache/{self.BASE_KEY}'
        self.model_save_dir = f'{cache_dir}/model_save/{self.BASE_KEY}'
        self.log_save_dir = f'{cache_dir}/log/{self.BASE_KEY}'

        # self.optimizer = torch.optim.SGD(self.gather_all_param(*self.models, self.loss_func), momentum=0.9, weight_decay=1e-5,lr=lr)
        self.optimizer = torch.optim.Adam(self.gather_all_param(*self.models, self.loss_func), lr=lr)
        self.log_name_key = log_name_key

        for model in models + [loss_func]:
            print(model.name, 'size', cal_model_size(model), 'MB')

    def train(self, start=-1):
        """
        Finish the full training process.

        :param start: if given a value of 0 or higher, will try to load the trained model 
            cached after the start-th epoch training, and resume training from there.
        """
        train_logs = self.train_epoches(start)
        self.save_models()

        # Save training logs to a HDF5 file.
        create_if_noexists(self.log_save_dir)
        train_logs.to_hdf(f'{self.log_save_dir}/{self.log_name_key}.h5', key='pretrain_log')

        # Report final result if use_nni
        if self.use_nni:
            nni.report_final_result(float(train_logs['loss'].to_list()[-1]))

    def train_epoches(self, start=-1, desc='Pre-training'):
        """ Train the models for multiple iterations (denoted by num_epoch). """
        self.train_state()

        if start > -1:
            self.load_models(start)
            print('Resumed training from epoch', start)

        train_logs = []
        desc_text = f'{desc}, avg loss %.4f'
        with trange(start+1, self.num_epoch, desc=desc_text % 0.0, disable=self.disable_tqdm) as tbar:
            for epoch_i in tbar:
                s_time = time()
                epoch_avg_loss = self.train_epoch(epoch_i)
                e_time = time()
                tbar.set_description(desc_text % epoch_avg_loss)
                train_logs.append([epoch_i, e_time - s_time, epoch_avg_loss])
                # Report intermediate result if use_nni
                if self.use_nni:
                    nni.report_intermediate_result(float(epoch_avg_loss))

                if self.cache_epoches and epoch_i < self.num_epoch - 1:
                    self.save_models(epoch_i)

        train_logs = pd.DataFrame(train_logs, columns=['epoch', 'time', 'loss'])
        return train_logs

    def train_epoch(self, epoch_i=None):
        """ Train the models for one epoch. """
        loss_log = []
        for batch_meta in tqdm(self.dataloader,
                               desc=f'-->Traverse batches', total=len(self.dataloader), leave=False,
                               disable=self.disable_tqdm):
            batch_meta = [e.to(self.device) if isinstance(e, torch.Tensor) else e for e in batch_meta]
            self.optimizer.zero_grad()
            loss = self.forward_loss(batch_meta)
            loss.backward()
            self.optimizer.step()

            loss_log.append(loss.item())
        return float(np.mean(loss_log))

    def finetune(self, **ft_params):
        for key, value in ft_params.items():
            if key in self.__dict__:
                setattr(self, key, value)
        self.prepare_batch_iter(0)
        train_logs = self.train_epoches(desc='Fine-tuning')

        create_if_noexists(self.log_save_dir)
        train_logs.to_hdf(f'{self.log_save_dir}/{self.log_name_key}.h5', key='finetune_log')

    @abstractmethod
    def forward_loss(self, batch_meta):
        """
        Controls how the trainer forward models and meta datas to the loss function.
        Might be different depending on specific type of pretex task.
        """
        return self.loss_func(self.models, *batch_meta)

    @staticmethod
    def gather_all_param(*models):
        """ Gather all learnable parameters in the models as a list. """
        parameters = []
        for encoder in models:
            parameters += list(encoder.parameters())
        return parameters

    def save_models(self, epoch=None):
        """ Save learnable parameters in the models as pytorch binaries. """
        for model in (*self.models, self.loss_func):
            if epoch is not None:
                create_if_noexists(self.model_cache_dir)
                save_path = f'{self.model_cache_dir}/{model.name}_epoch{epoch}.model'
            else:
                create_if_noexists(self.model_save_dir)
                save_path = f'{self.model_save_dir}/{model.name}.model'
                print('Saved model', model.name)
            torch.save(model.state_dict(), save_path)

    def load_model(self, model, epoch=None):
        """ Load one of the model. """
        if epoch is not None:
            save_path = f'{self.model_cache_dir}/{model.name}_epoch{epoch}.model'
        else:
            save_path = f'{self.model_save_dir}/{model.name}.model'
        model.load_state_dict(torch.load(save_path, map_location=self.device))
        print('Load model', model.name)
        return model

    def load_models(self, epoch=None):
        """ 
        Load all models from file. 
        """
        for i, model in enumerate(self.models):
            self.models[i] = self.load_model(model, epoch)
        self.loss_func = self.load_model(self.loss_func, epoch)

    def get_models(self):
        """ Obtain all models in the trainer in evluation state. """
        self.eval_state()
        return self.models

    def train_state(self):
        for model in self.models:
            model.train()
        self.loss_func.train()

    def eval_state(self):
        for model in self.models:
            model.eval()
        self.loss_func.eval()


class ContrastiveTrainer(Trainer):
    """
    Trainer for contrastive pre-training.
    """

    def __init__(self, **kwargs):
        """
        :param contra_meta_i: list of meta indices indicating which meta data to use for constrative training.
            The indices corresponds to the meta_types list.
        """
        super().__init__(trainer_name='contrastive',
                         **kwargs)

    def forward_loss(self, batch_meta):
        return self.loss_func(self.models, *batch_meta)


class GenerativeTrainer(Trainer):
    """
    Trainer for generative pre-training.
    Contains a generate function for evaluating the recovered input.
    """

    def __init__(self, **kwargs):
        """
        :param enc_meta_i: list of meta indices indicating which meta data to fed into the encoders.
        :param rec_meta_i: list of meta indices indicating which meta data to use as recovery target.
        """
        super().__init__(trainer_name='generative',
                         **kwargs)
        self.generation_save_dir = f'{self.cache_dir}/generation/{self.BASE_KEY}'

    def forward_loss(self, batch_meta):
        """ For generative training, the batch is split into encode and recovery meta, then fed into the loss function. """
        return self.loss_func(self.models, *batch_meta)

    def generate(self, set_index, **gen_params):
        """ Generate and save recovered meta data. """
        for key, value in gen_params.items():
            if key in self.__dict__:
                setattr(self, key, value)

        raise NotImplementedError("Not implemented yet")


class MomentumTrainer(Trainer):
    """
    Trainer for momentum-style parameter updating.
    Requires the loss function contains extra "teacher" models symmetric to the base models.
    The parameters of the teacher models will be updated in a momentum-style.
    """

    def __init__(self, momentum, teacher_momentum, weight_decay, eps, warmup_epoch=10, **kwargs):
        super().__init__(trainer_name='momentum',
                         suffix=f'_m{momentum}-tm{teacher_momentum}-wd{weight_decay}-eps{eps}-we{warmup_epoch}',
                         **kwargs)

        self.momentum = momentum
        self.teacher_momentum = teacher_momentum
        self.warmup_epoch = warmup_epoch
        self.lamda = 1 / (kwargs['batch_size'] * eps / self.models[0].output_size)

        self.optimizer = torch.optim.SGD(self.gather_all_param(*self.models, self.loss_func), lr=self.lr,
                                         momentum=momentum, weight_decay=weight_decay)

    def train(self, start=-1):
        self.prepare_batch_iter(0)
        # The schedules are used for controlling the learning rate, momentum, and lamda.
        self.momentum_schedule = self.cosine_scheduler(self.teacher_momentum, 1,
                                                       self.num_epoch, self.num_iter)
        self.lr_schedule = self.cosine_scheduler(self.lr, 0, self.num_epoch,
                                                 self.num_iter, warmup_epochs=self.warmup_epoch)
        self.lamda_schedule = self.lamda_scheduler(8/self.lamda, 1/self.lamda, self.num_epoch, self.num_iter,
                                                   warmup_epochs=self.warmup_epoch)
        train_logs = self.train_epoches(start)

        self.save_models()

        # Save training logs to a HDF5 file.
        create_if_noexists(self.log_save_dir)
        train_logs.to_hdf(f'{self.log_save_dir}/{self.log_name_key}.h5', key='pretrain_log')

    def train_epoch(self, epoch_i):
        loss_log = []
        for batch_i, batch_meta in tqdm(enumerate(self.dataloader),
                                        desc=f'{self.trainer_name} training {epoch_i+1}-th epoch',
                                        total=len(self.dataloader), disable=self.disable_tqdm):
            it = self.num_iter * epoch_i + batch_i
            cur_lr = self.lr_schedule[it]
            lamda_inv = self.lamda_schedule[it]
            momentum = self.momentum_schedule[it]

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = cur_lr

            self.optimizer.zero_grad()
            batch_meta = [e.to(self.device) for e in batch_meta]
            loss = self.loss_func(self.models, *batch_meta,
                                  lamda_inv=lamda_inv, momentum=momentum)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                if self.loss_func[0].teachers is None:
                    models_len = len(self.models)
                    for encoder, teacher in zip(self.models[:models_len//2], self.models[models_len//2:]):
                        for param_q, param_k in zip(encoder.parameters(), teacher.parameters()):
                            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)
                else:
                    for encoder, teacher in zip(self.models, self.loss_func[0].teachers):
                        for param_q, param_k in zip(encoder.parameters(), teacher.parameters()):
                            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

            loss_log.append(loss.item())
        return float(np.mean(loss_log))

    @staticmethod
    def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule

    @staticmethod
    def lamda_scheduler(start_warmup_value, base_value, epochs, niter_per_ep, warmup_epochs=5):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        schedule = np.ones(epochs * niter_per_ep - warmup_iters) * base_value
        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule


class NoneTrainer():
    def __init__(self, models, data, cache_dir, device):
        self.models = [model.to(device) for model in models]
        self.BASE_KEY = f'end2end/none/{data.name}'
        self.device = device
        self.model_save_dir = f'{cache_dir}/model_save/{self.BASE_KEY}'

    def save_models(self):
        """ Save all models. """
        create_if_noexists(self.model_save_dir)
        for model in self.models:
            save_path = f'{self.model_save_dir}/{model.name}.model'
            torch.save(model.state_dict(), save_path)

    def load_model(self, model):
        """ Load one of the encoder. """
        save_path = f'{self.model_save_dir}/{model.name}.model'
        model.load_state_dict(torch.load(save_path, map_location=self.device))
        print('Load model from', save_path)
        return model

    def load_models(self):
        """ Load all encoders. """
        for i, model in enumerate(self.models):
            self.models[i] = self.load_model(model)

    def get_models(self):
        for model in self.models:
            model.eval()
        return self.models
