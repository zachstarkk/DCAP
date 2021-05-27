import math
import shutil
import struct
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import lmdb
import numpy as np
import torch.utils.data
from tqdm import tqdm


class RapidAdvanceDataset(torch.utils.data.Dataset):
    """
    MovieLens 1M Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    :param dataset_path: MovieLens dataset path

    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path):
        super().__init__()
        ratings_info = pd.read_csv(dataset_path+'/train.csv', sep='::', engine='python')
        self.items = ratings_info.iloc[:, 1:]  # -1 because ID begins from 1
        self.targets = ratings_info.iloc[:, 0].to_numpy().astype(np.float32)
        
        self.items['1_y'][self.items['1_y']=='M'] = 0
        self.items['1_y'][self.items['1_y']=='F'] = 1
        self.items = self.items.to_numpy().astype(np.int)
        self.targets = self.__preprocess_target(self.targets).astype(np.float32)
        
        self.field_dims = np.max(self.items, axis=0) + 1
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]
        
    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target

    def __preprocess_items(self, items):
        for i in range(15):
            items.iloc[:, i] = 


@lru_cache(maxsize=None)
def convert_numeric_feature(val: str):
    if val == '':
        return 'NULL'
    v = int(val)
    if v > 2:
        return str(int(math.log(v) ** 2))
    else:
        return str(v - 2)