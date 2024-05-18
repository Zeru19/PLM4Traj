import copy
import os
import math
import random
from collections import Counter, defaultdict
from itertools import islice, zip_longest
from time import time
import platform
import json
import gc
import h5py
from copy import deepcopy
from itertools import combinations
from scipy.spatial.distance import cdist

import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from einops import repeat, rearrange, reduce
from scipy import sparse, spatial
from tqdm import tqdm, trange
import networkx as nx
from sklearn.utils import shuffle
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.metrics.pairwise import euclidean_distances

from utils import create_if_noexists, remove_if_exists, geo_distance, next_batch, calc_azimuth

pd.options.mode.chained_assignment = None
CLASS_COL = 'driver'
SET_NAMES = [(0, 'train'), (1, 'val'), (2, 'test')]
MIN_TRIP_LEN = 6
MAX_TRIP_LEN = 120
TARGET_SAMPLE_RATE = 15
TRIP_COLS = ['tod', 'road', 'road_prop', 'lng', 'lat', 'weekday', 'seq_i', 'seconds', 
             'speed', 'acceleration', 'heading_angle']

with open('path_conf.json') as fp:
    conf = json.load(fp)


class Data:
    def __init__(self, name, road_type='road_network', **kwargs):
        self.name = name
        self.small = 'small' in name

        paths = conf['small'] if self.small else conf['full']
        if kwargs.get('use_nni', False):
            paths = conf['nni']
        self.base_path = paths['meta_path']
        self.dataset_path = paths['dataset_path']

        self.df_path = f'{self.dataset_path}/{self.name}.h5'
        self.meta_dir = f'{self.base_path}/meta/{self.name}'
        self.stat_path = f'{self.meta_dir}/stat_grid.h5' if road_type == 'grid' else f'{self.meta_dir}/stat.h5'

        self.get_meta_path = lambda meta_type, select_set: os.path.join(
            self.meta_dir, f'{meta_type}_' + \
                           ('grid_' if road_type == 'grid' else '') + \
                           f'{select_set}.npz'
        )

        assert road_type in ['road_network', 'grid'], "road type must be road_network or grid"
        self.road_type = road_type

    """ Load functions for loading dataframes and meta. """

    def read_hdf(self):
        # Load the raw data from HDF files.
        # One set of raw dataset is composed of one HDF file with four keys.
        # The trips contains the sequences of trajectories, with three columns: trip, time, road
        self.trips = pd.read_hdf(self.df_path, key='trips')
        # The trip_info contains meta information about trips. For now, this is mainly used for class labels.
        self.trip_info = pd.read_hdf(self.df_path, key='trip_info')
        # The road_info contains meta information about roads.
        self.road_info = pd.read_hdf(self.df_path, key='road_info')
        # self.trips = pd.merge(self.trips, self.road_info[['road', 'lng', 'lat']], on='road', how='left')
        self.network_info = None
        self.network = None

        if self.road_type == 'grid':
            self.project_to_grid()

        # Add some columns to the trip
        self.trips = self.trips.reset_index(drop=True)
        self.trips['seconds'] = self.trips['time'].apply(lambda x: x.timestamp())
        self.trips['tod'] = self.trips['seconds'] % (24 * 60 * 60) / (24 * 60 * 60)
        self.trips['weekday'] = self.trips['time'].dt.weekday
        self.trips['speed'] = 0.
        self.trips['acceleration'] = 0.
        self.trips['heading_angle'] = 0.
        self.stat = self.trips.describe()

        num_road = int(self.road_info['road'].max() + 1)
        num_class = int(self.trip_info[CLASS_COL].max() + 1)
        if self.road_type == 'grid':
            self.data_info = pd.Series([num_road, num_class, self.num_w, self.num_h],
                                       index=['num_road', 'num_class', 'num_w', 'num_h'])
        else:
            self.data_info = pd.Series([num_road, num_class], index=['num_road', 'num_class'])
        print('Loaded DataFrame from', self.df_path)
        self.num_road = num_road

        num_trips = self.trip_info.shape[0]
        self.train_val_test_trips = (self.trip_info['trip'].iloc[:int(num_trips * 0.8)],
                                     self.trip_info['trip'].iloc[int(num_trips * 0.8):int(num_trips * 0.9)],
                                     self.trip_info['trip'].iloc[int(num_trips * 0.9):])

        create_if_noexists(self.meta_dir)
        self.stat.to_hdf(self.stat_path, key='stat')
        self.data_info.to_hdf(self.stat_path, key='info')
        print(self.data_info)
        print('Dumped dataset info into', self.stat_path)

        self.valid_trips = [self.get_valid_trip_id(i) for i in range(3)]

    @staticmethod
    def compute_speed_acc_angle(trips: pd.DataFrame):
        MAX_SPEED = 33.33
        MIN_SPEED = -33.33
        MIN_ACCELERATION = -3
        MAX_ACCELERATION = 3

        trips['speed'] = 0.
        trips['acceleration'] = 0.
        trips['heading_angle'] = 0.

        print("Compute speed and acceleration, ", end='')
        for _, group in tqdm(trips.groupby('trip'), desc='Computing speed and acceleration',
                             total=trips['trip'].nunique()):
            speed = np.zeros_like(group['seconds'])
            acceleration = np.zeros_like(group['seconds'])
            heading_angle = np.zeros_like(group['seconds'])
            for i in range(1, len(group)):
                time_diff = group.iloc[i]['seconds'] - group.iloc[i - 1]['seconds']
                if time_diff > 0:
                    speed[i] = geo_distance(group.iloc[i]['lng'], group.iloc[i]['lat'],
                                            group.iloc[i - 1]['lng'], group.iloc[i - 1]['lat']) / time_diff
                    # Truncate speed to its range
                    speed[i] = max(min(speed[i], MAX_SPEED), MIN_SPEED)

                    acceleration[i] = (speed[i] - speed[i - 1]) / time_diff
                    # Truncate acceleration to its range
                    acceleration[i] = max(min(acceleration[i], MAX_ACCELERATION), MIN_ACCELERATION)

                heading_angle[i] = calc_azimuth(group.iloc[i]['lng'], group.iloc[i]['lat'],
                                                group.iloc[i - 1]['lng'], group.iloc[i - 1]['lat'])
            heading_angle[0] = heading_angle[1]  # set the heading angle of the first point with the second point
            trips.loc[group.index, 'speed'] = speed
            trips.loc[group.index, 'acceleration'] = acceleration
            trips.loc[group.index, 'heading_angle'] = heading_angle
        return trips

    def compute_speed_acc_angle_of_all_trips(self):
        self.trips = self.compute_speed_acc_angle(self.trips)
        self.stat = self.trips.describe()
        self.stat.to_hdf(self.stat_path, key='expanded_stat')
        print("Done.")

    def project_to_grid(self):
        print("Project trips to grids, ", end='')
        delta_degree = 0.0015
        min_lng, max_lng = self.trips['lng'].min(), self.trips['lng'].max()
        min_lat, max_lat = self.trips['lat'].min(), self.trips['lat'].max()
        num_h = math.ceil((max_lat - min_lat) / delta_degree)
        num_w = math.ceil((max_lng - min_lng) / delta_degree)
        self.num_h = num_h
        self.num_w = num_w
        self.num_road = num_w * num_h

        def _project_lng_to_w(lng):
            return np.clip(np.ceil((lng - min_lng) / (max_lng - min_lng) * num_w) - 1, 0,
                           num_w - 1)

        def _project_lat_to_h(lat):
            return np.clip(np.ceil((lat - min_lat) / (max_lat - min_lat) * num_h) - 1, 0,
                           num_h - 1)

        ws = self.trips['lng'].apply(_project_lng_to_w)
        hs = self.trips['lat'].apply(_project_lat_to_h)
        self.trips['road'] = [h * num_w + w for h, w in zip(hs, ws)]

        road_lngs = np.arange(num_w) * delta_degree + delta_degree / 2 + min_lng
        road_lats = np.arange(num_h) * delta_degree + delta_degree / 2 + min_lat
        road_lngs = repeat(road_lngs, 'W -> (H W)', H=num_h)
        road_lats = repeat(road_lats, 'H -> (H W)', W=num_w)

        self.road_info = pd.DataFrame({
            "road": list(range(num_w * num_h)),
            "lng": road_lngs,
            "lat": road_lats
        })

        print("None.")

    def load_stat(self):
        # Load statistical information for features.
        self.stat = pd.read_hdf(self.stat_path, key='stat')
        self.data_info = pd.read_hdf(self.stat_path, key='info')
        # if expanded_stat in stat hdf, load it
        if 'expanded_stat' in h5py.File(self.stat_path, 'r').keys():
            print("load expanded_stat from stat hdf")
            self.stat = pd.read_hdf(self.stat_path, key='expanded_stat')
        print(self.stat)

    def load_meta(self, meta_type, select_set):
        meta_path = self.get_meta_path(meta_type, select_set)
        loaded = np.load(meta_path, allow_pickle=True)
        # print('Loaded meta from', meta_path)
        return list(loaded.values())

    def get_valid_trip_id(self, select_set):
        select_trip_id = self.train_val_test_trips[select_set]
        trips = self.trips[self.trips['trip'].isin(select_trip_id)]
        valid_trip_id = []
        for _, group in tqdm(trips.groupby('trip'), desc='Filtering trips', total=select_trip_id.shape[0], leave=False):
            if (not group.isna().any().any()) and group.shape[0] >= MIN_TRIP_LEN and group.shape[0] <= MAX_TRIP_LEN:
                # if ((group['seconds'] - group.shift(1)['seconds']).iloc[1:] == TARGET_SAMPLE_RATE).all():  # using in dataset made by tonglong
                valid_trip_id.append(group.iloc[0]['trip'])
        return valid_trip_id

    def dump_meta(self, meta_type, select_set):
        """
        Dump meta data into numpy binary files for fast loading later.

        :param meta_type: type name of meta data to dump.
        :param select_set: index of set to dump. 0 - training set, 1 - validation set, and 2 - testing set.
        """
        # Prepare some common objects that will probably be useful for various types of meta data.
        cal_acc = False
        if 'expanded_stat' not in h5py.File(self.stat_path, 'r').keys():
            self.compute_speed_acc_angle_of_all_trips()
            cal_acc = True
        self.load_stat()
        select_trip_id = self.valid_trips[select_set]
        known_trips = self.trips[self.trips['trip'].isin(self.valid_trips[0] + self.valid_trips[1])]
        trips = self.trips[self.trips['trip'].isin(select_trip_id)]
        trip_info = self.trip_info[self.trip_info['trip'].isin(select_trip_id)]
        max_trip_len = max(Counter(trips['trip']).values())
        trip_normalizer = Normalizer(self.stat, feat_cols=[0, 3, 4, 8, 9, 10], norm_type='minmax')

        if meta_type == 'trip':
            """
            The "trip" meta data obeys the original form of trajectories. 
            One complete trajectory sequence is regarded as one trip.
            """
            if not cal_acc:
                trips = self.compute_speed_acc_angle(trips)

            arrs, valid_lens = [], []
            for _, group in tqdm(trips.groupby('trip'), desc='Gathering trips', total=len(select_trip_id)):
                arr = group[TRIP_COLS].to_numpy()
                # fill nan
                arr = np.nan_to_num(arr, nan=0)

                offset = group['time'].apply(lambda x: x.timestamp()).to_numpy()
                offset = (offset - offset[0]) / (offset[-1] - offset[0]) * 2 - 1
                arr = np.append(arr, offset.reshape(-1, 1), 1)
                arr = trip_normalizer(arr)

                arrs.append(arr)
            # Features of arrs: TRIP_COLS + [offset]
            arrs = np.array(arrs, dtype=object)
            
            meta = [arrs]

        elif 'odpois' in meta_type:
            """ POIs near the o and d of each trip. """
            params = meta_type.split('-')
            max_pois = int(params[1]) if len(params) > 1 else 5
            pre_length = int(params[2]) if len(params) > 2 else 0
            momentum_extra = bool(int(params[3])) if len(params) > 3 else False

            def _extrapolate_lnglat(df, step):
                if step < 1:
                    return df
                
                # Extrapolate the lng and lat of the last point of the trip.
                # Calculate speed of longitude and latitude dimension.
                time_gap = df.iloc[-1]['seconds'] - df.iloc[-2]['seconds']
                step = np.arange(1, step + 1)

                lng_speed = (df.iloc[-1]['lng'] - df.iloc[-2]['lng']) / time_gap
                lat_speed = (df.iloc[-1]['lat'] - df.iloc[-2]['lat']) / time_gap
                # Extrapolate the lng and lat of the last point, by step * the last time gap.
                lng = df.iloc[-1]['lng'] + step * lng_speed
                lat = df.iloc[-1]['lat'] + step * lat_speed
                seconds = df.iloc[-1]['seconds'] + step * time_gap
                
                res_df = pd.DataFrame({'lng': lng, 'lat': lat, 'seconds': seconds})
                res_df = pd.concat([df, res_df], ignore_index=True)

                return res_df

            pois = pd.read_hdf(self.df_path, key='pois')
            pois = pois[['name', 'address', 'lng', 'lat']].to_numpy()
            knn = NearestNeighbors(n_neighbors=max_pois, metric='euclidean')
            knn.fit(pois[:, 2:])

            o_poi_list, d_poi_list = [], []
            for _, group in tqdm(trips.groupby('trip'), desc='Gathering od pois', total=len(select_trip_id)):
                o_lng, o_lat = group['lng'].to_list()[0], group['lat'].to_list()[0]
                d_df = _extrapolate_lnglat(group.iloc[:-1-pre_length], pre_length) if momentum_extra else group.iloc[:-1-pre_length]
                d_lng, d_lat = d_df['lng'].to_list()[-1], d_df['lat'].to_list()[-1]
                o_pois = knn.kneighbors([[o_lng, o_lat]], return_distance=False)[0]
                d_pois = knn.kneighbors([[d_lng, d_lat]], return_distance=False)[0]
                # concat directly
                # o_pois = [e[1] + e[0] for e in pois[o_pois].tolist()]
                # d_pois = [e[1] + e[0] for e in pois[d_pois].tolist()]
                
                # one address, many pois
                o_descs = [e[0] for e in pois[o_pois].tolist()]
                d_descs = [e[0] for e in pois[d_pois].tolist()]
                o_pois = [pois[o_pois[0]][1] + '、'.join(o_descs)] 
                d_pois = [pois[d_pois[0]][1] + '、'.join(d_descs)]

                o_poi_list.append(o_pois)
                d_poi_list.append(d_pois)
            
            meta = [o_poi_list, d_poi_list]

        elif 'destination' in meta_type:
            params = meta_type.split('-')

            try:
                trip, = self.load_meta('trip', select_set)
            except FileNotFoundError:
                self.dump_meta('trip', select_set)
                trip = self.load_meta('trip', select_set)
            
            dests = []
            for arr in tqdm(trip, desc='Gathering destinations', total=len(trip)):
                dests.append(arr[-1, 1])
            dests = np.array(dests).astype(int)

            meta = [dests]

        elif 'tte' in meta_type:
            params = meta_type.split('-')

            try:
                trip, = self.load_meta('trip', select_set)
            except FileExistsError:
                self.dump_meta('trip', select_set)
                trip = self.load_meta('trip', select_set)

            ttes = []
            for arr in tqdm(trip, desc='Gathering ttes', total=len(trip)):
                # ttes.append((arr[-1, 7] - arr[0, 7]) / 60)  # minutes
                ttes.append((arr[-1, 7] - arr[0, 7]))  # seconds
            ttes = np.array(ttes)

            meta = [ttes]

        else:
            raise NotImplementedError('No meta type', meta_type)

        create_if_noexists(self.meta_dir)
        meta_path = self.get_meta_path(meta_type, select_set)
        np.savez(meta_path, *meta)
        print('Saved meta to', meta_path)


class Normalizer(nn.Module):
    def __init__(self, stat, feat_cols, feat_names=None, norm_type='zscore'):
        super().__init__()

        self.stat = stat
        self.feat_cols = feat_cols
        self.feat_names = feat_names if feat_names is not None \
            else [TRIP_COLS[feat_col] for feat_col in feat_cols]
        self.norm_type = norm_type

    def _norm_col(self, x_col, col_name):
        if self.norm_type == 'zscore':
            x_col = (x_col - self.stat.loc['mean', col_name]) / self.stat.loc['std', col_name]
        elif self.norm_type == 'minmax':
            x_col = (x_col - self.stat.loc['min', col_name]) / \
                    (self.stat.loc['max', col_name] - self.stat.loc['min', col_name])
            x_col = x_col * 2 - 1
        else:
            raise NotImplementedError(self.norm_type)
        return x_col

    def forward(self, arr):
        """ Normalize the input array. """
        if isinstance(arr, torch.Tensor):
            x = torch.clone(arr)
        else:
            x = np.copy(arr)
        for col, name in zip(self.feat_cols, self.feat_names):
            x[..., col] = self._norm_col(x[..., col], name)
        return x


class Denormalizer(nn.Module):
    def __init__(self, stat, feat_cols, feat_names=None, norm_type='zscore'):
        super().__init__()

        self.stat = stat
        self.feat_names = feat_names
        self.feat_cols = feat_cols
        self.feat_names = feat_names if feat_names is not None \
            else [TRIP_COLS[feat_col] for feat_col in feat_cols]
        self.norm_type = norm_type

    def _denorm_col(self, x_col, col_name):
        if self.norm_type == 'zscore':
            x_col = x_col * self.stat.loc['std', col_name] + self.stat.loc['mean', col_name]
        elif self.norm_type == 'minmax':
            x_col = (x_col + 1) / 2
            x_col = x_col * (self.stat.loc['max', col_name] - self.stat.loc['min', col_name]) + \
                    self.stat.loc['min', col_name]
        else:
            raise NotImplementedError(self.norm_type)
        return x_col

    def forward(self, select_cols, arr):
        """ Denormalize the input batch. """
        if isinstance(arr, torch.Tensor):
            x = torch.clone(arr)
        else:
            x = np.copy(arr)
        for col, name in zip(self.feat_cols, self.feat_names):
            if col in select_cols:
                x[..., col] = self._denorm_col(x[..., col], name)
        return x


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-n', '--name', help='the name of the dataset', type=str, default="small_chengdu") 
    parser.add_argument('-t', '--types', help='the type of meta data to dump', type=str, default="trip")
    parser.add_argument('-i', '--indices', help='the set index to dump meta', type=str, default="0,1,2")
    parser.add_argument('-g', '--grid', action='store_true', help="whether to project to grids.")

    args = parser.parse_args()

    road_type = 'grid' if args.grid else 'road_network'
    data = Data(args.name, road_type)
    data.read_hdf()
    for type in args.types.split(','):
        for i in args.indices.split(','):
            data.dump_meta(type, int(i))
            # Test if we can load meta from the file
            meta = data.load_meta(type, i)
