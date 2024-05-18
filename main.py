import json
import os
from argparse import ArgumentParser
import copy
from functools import partial

import torch
from torch.cuda import is_available as cuda_available
from torch.utils.data import DataLoader

from data import Data
from pretrain import trainer as PreTrainer
from loss import TripCausalLoss
from model import LET
from dataloader import TripODPOIWithHour
from downstream import task, predictor as DownPredictor
import utils

os.environ["TOKENIZERS_PARALLELISM"] = "true"


# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('-c', '--config', help='name of the config file to use', type=str, default="small_chengdu")
parser.add_argument('--cuda', help='index of the cuda device to use', type=int, default=0)
parser.add_argument('--use-nni', help='whether to use nni', action='store_true')
args = parser.parse_args()
device = f'cuda:{int(args.cuda)}' if cuda_available() else 'cpu'
datetime_key = utils.get_datetime_key()
pretrainer_load_epoch = None

# NNI on config file
# If use_nni, load the config params
use_nni = bool(args.use_nni)
if use_nni:
    import nni
    nni_params = nni.get_next_parameter()
    if 'config' in nni_params:
        args.config = nni_params['config']

# Load config file
with open(f'config/{args.config}.json', 'r') as fp:
    config = json.load(fp)

dataloader_map = {
    'trip_with_odpoi_hour': TripODPOIWithHour
}

# Each config file can contain multiple entries. Each entry is a different set of configuration.
for num_entry, entry in enumerate(config):
    print(f'\n{"=" * 30}\n===={num_entry+1}/{len(config)} experiment entry====')

    # Load dataset.
    data_entry = entry['data']
    data = Data(data_entry['name'], data_entry.get('road_type', 'road_network'), use_nni=use_nni)
    data.load_stat()
    num_roads = data.data_info['num_road']

    conf_save_dir = os.path.join(data.base_path, 'config')
    utils.create_if_noexists(conf_save_dir)
    with open(os.path.join(conf_save_dir, f'{datetime_key}_e{num_entry}.json'), 'w') as fp:
        json.dump(entry, fp)

    # Each entry can be repeated for several times.
    num_repeat = entry.get('repeat', 1)
    for repeat_i in range(num_repeat):
        print(f'\n----{num_entry+1}/{len(config)} experiment entry, {repeat_i+1}/{num_repeat} repeat----\n')

        models = []
        for model_entry in entry['models']:
            # Create models.
            model_name = model_entry['name']
            model_config = model_entry.get('config', {})
            if "pre_embed" in model_config:
                model_config["pre_embed"] = data.load_meta(model_config.get("pre_embed"), 0)[0]
                model_config["pre_embed_update"] = model_config.get("pre_embed_update", True)

            if model_name == 'let':
                models.append(LET(**model_config))
            else:
                raise NotImplementedError(f'Unknown model name {model_name}')

        if 'pretrain' in entry:
            # Create pre-training loss function.
            pretrain_entry = entry['pretrain']
            loss_entry = pretrain_entry['loss']
            loss_name = loss_entry['name']

            loss_param = loss_entry.get('config', {})

            if loss_name == 'trip_causal':
                loss_func = TripCausalLoss(**loss_param)
            else:
                raise NotImplementedError(f"Unknow loss name {loss_name}")
            
            print(loss_func.name, 'size', utils.cal_model_size(loss_func), 'MB')

            # Create pre-trainer.
            pretrainer_entry = pretrain_entry['trainer']
            pretrainer_name = pretrainer_entry['name']

            # Prepare dataloader for the trainer
            dataloader_entry = pretrain_entry['dataloader']
            dataloader_name = dataloader_entry['name']
            if dataloader_name in dataloader_map:
                DatasetClass = dataloader_map[dataloader_name]
            else:
                raise NotImplementedError(f'Unknown dataloader name {dataloader_name}')

            dataset_config = dataloader_entry.get('dataset_config', {})
            dataloader_config = dataloader_entry.get('config', {})
            dataloader_collate_fn_config = dataloader_entry.get('collate_fn_config', {})
            meta_types = dataloader_entry.get('meta_types', ['trip'])
            pretrain_data_name = '_'.join([data.name] + meta_types)

            metas = []
            for meta_type in meta_types:
                metas = metas + data.load_meta(meta_type, 0)
            pretrain_dataset = DatasetClass(*metas, **dataset_config)
            pretrain_dataloader = DataLoader(pretrain_dataset,
                                             collate_fn=partial(DatasetClass.collate_fn, **dataloader_collate_fn_config),
                                             **dataloader_config)

            # TODO: Improve the process, move dataloader forward to the loss.
            if getattr(pretrain_dataset, 'road_dist', None) is not None and isinstance(loss_func, MLM):
                loss_func.road_dist = torch.from_numpy(pretrain_dataset.road_dist).float().to(device)

            # Prepare pretrainer config
            pretrainer_comm_param = {"dataloader": pretrain_dataloader, "meta_name": pretrain_data_name, "cache_dir": data.base_path,
                                     "models": models, "loss_func": loss_func, 
                                     "device": device, "log_name_key": datetime_key + f'_e{num_entry}_r{repeat_i}'}
            pretrainer_config = pretrainer_entry.get('config', {}) | pretrainer_comm_param
            
            if pretrainer_name == 'contrastive':
                pre_trainer = PreTrainer.ContrastiveTrainer(**pretrainer_config)
            elif pretrainer_name == 'generative':
                pre_trainer = PreTrainer.GenerativeTrainer(**pretrainer_config)
            else:
                raise NotImplementedError(f'Unknown pretrainer name {pretrainer_name}')
            
            # Pre-training on the training set, or load from trained cache.
            if pretrain_entry.get('load', False):
                if pretrain_entry.get('load_epoch', False):
                    pretrainer_load_epoch = int(pretrain_entry['load_epoch'])
                    pre_trainer.load_models(epoch=pretrainer_load_epoch)
                else:
                    pre_trainer.load_models()
            else:
                pre_trainer.train(pretrain_entry.get('resume', -1))

        else:
            pre_trainer = PreTrainer.NoneTrainer(models=models, data=data, device=device, cache_dir=data.base_path)
            # pre_trainer.save_models()
            print('Skip pretraining.')

        # Downstream evaluation
        if 'downstream' in entry:
            num_down = len(entry['downstream'])
            for down_i, down_entry in enumerate(entry['downstream']):
                print(f'\n....{num_entry+1}/{len(config)} experiment entry, {repeat_i+1}/{num_repeat} repeat, '
                      f'{down_i+1}/{num_down} downstream task ....\n')

                if down_i > 0:
                    if pretrainer_load_epoch is not None:
                        pre_trainer.load_models(epoch=pretrainer_load_epoch)
                    else:
                        pre_trainer.load_models()
                    models = pre_trainer.get_models()

                down_models = [models[i] for i in down_entry['select_models']]
                down_embed_size = sum([model.output_size for model in down_models])
                down_task = down_entry['task']

                # Dataset for downstream task
                dataloader_entry = down_entry.get('dataloader', {})
                dataloader_name = dataloader_entry['name']
                if dataloader_name in dataloader_map:
                    DatasetClass = dataloader_map[dataloader_name]
                else:
                    raise NotImplementedError(f'Unknown dataloader name {dataloader_name}')
                
                meta_types = dataloader_entry.get('meta_types', ['trip'])
                train_metas, eval_metas = [], []
                for meta_type in meta_types:
                    train_metas = train_metas + data.load_meta(meta_type, 0)
                    eval_metas = eval_metas + data.load_meta(meta_type, int(down_entry['eval_set']))
                dataset_config = dataloader_entry.get('dataset_config', {})
                dataloader_config = dataloader_entry.get('config', {})
                down_train_dataset = DatasetClass(*train_metas, **dataset_config)
                down_eval_dataset = DatasetClass(*eval_metas, **dataset_config)

                dataloader_collate_fn_config = dataloader_entry.get('collate_fn_config', {})
                down_train_dataloader = DataLoader(down_train_dataset,
                                                   collate_fn=partial(DatasetClass.collate_fn, **dataloader_collate_fn_config),
                                                   **dataloader_config)
                down_eval_dataloader = DataLoader(down_eval_dataset,
                                                  collate_fn=partial(DatasetClass.collate_fn, **dataloader_collate_fn_config),
                                                  **dataloader_config)

                down_comm_params = {
                    "train_data": down_train_dataloader, "eval_data": down_eval_dataloader,
                    "models": down_models, "device": device, "cache_dir": data.base_path,
                    "base_name": pre_trainer.BASE_KEY, "log_name_key": datetime_key + f'_e{num_entry}_r{repeat_i}',
                    "use_nni": use_nni
                }
                down_config = down_entry.get('config', {}) | down_comm_params
                
                predictor_entry = down_entry.get('predictor', {}) 
                predictor_config = predictor_entry.get('config', {}) 
                if down_task == 'destination':
                    predictor = DownPredictor.FCPredictor(
                        input_size=down_embed_size, output_size=num_roads,
                        **predictor_config)
                    down_trainer = task.Destination(
                        predictor=predictor,
                        **down_config)
                elif down_task == 'tte':
                    predictor = DownPredictor.FCPredictor(
                        input_size=down_embed_size, output_size=1,
                        **predictor_config)
                    down_trainer = task.TTE(
                        predictor=predictor,
                        **down_config)
                else:
                    raise NotImplementedError(f'No downstream task called "{down_task}".')

                if down_entry.get('load', False):
                    down_trainer.load_models()
                else:
                    down_trainer.train()
                down_trainer.eval(int(down_entry['eval_set']), full_metric=True)
        else:
            print('Finishing program without performing downstream tasks.')
