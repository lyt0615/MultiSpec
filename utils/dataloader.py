#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :dataloader.py
@Description :
@InitTime    :2024/05/10 10:23:04
@Author      :XinyuLu
@EMail       :xinyulu@stu.xmu.edu.cn
'''


from torch.utils.data import Dataset
from functools import lru_cache
import pickle
import lmdb
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
# from gensim.models import Word2Vec
from torch.utils.data.distributed import DistributedSampler
# from transformers import RobertaTokenizer


class lmdbDataset(Dataset):
    def __init__(self, lmdb_path, target_keys, device):
        self.lmdb_path = lmdb_path
        self.target_keys = target_keys
        self.device = device

        assert os.path.isfile(
            self.lmdb_path), "{} not found".format(self.lmdb_path)

        self.env = self._connect_db()
        with self.env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def _connect_db(self):
        env = lmdb.open(
            self.lmdb_path,
            subdir=False, readonly=True,
            lock=False, readahead=False,
            meminit=False, max_readers=256
        )
        return env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self._connect_db(self.lmdb_path, save_to_self=True)
        key = self._keys[idx]
        pickled_data = self.env.begin().get(key)
        data = pickle.loads(pickled_data)
        output = {}
        for k in self.target_keys:
            if k == 'smiles':
                output[k] = data[k]
            else:
                output[k] = torch.as_tensor(data[k])
        return output

class Collator:
    def __init__(self, task, word2vec_model=None, tokenizer_path=None):
        
        self.task = task
        self.word2vec_model = word2vec_model
        self.smiles_tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path) if tokenizer_path is not None else None

    def spectra_substructures_fn(self, batch):
        spectra = torch.stack([item['spectra'] for item in batch]).unsqueeze(1).to(torch.float32)
        substructures = torch.stack([item['substructures'] for item in batch]).to(torch.float32)
        return {'target':substructures, 'data': spectra}


    def __call__(self, batch):
        collate_fn_dict = {
            "spectra_substructures": self.spectra_substructures_fn,
            }

        return collate_fn_dict[self.task](batch)
    
def make_dataloader(lmdb_path, task, mode='train', word2vec=False,
                    batch_size=16, num_workers=0, device='cpu', ddp=False, verbose=False):

    if verbose: 
        print(f'[train set] = {lmdb_path} | [task] = {task}')

    target_keys = task.split("_") #
    if 'peaks' in target_keys:
        target_keys += ['mus', 'sigmas', 'amps', 'weights']
        target_keys.pop(target_keys.index('peaks'))

    dataset = lmdbDataset(f'/data/YantiLiu/projects/multispec/datasets/{lmdb_path}/{lmdb_path}_{mode}.lmdb', target_keys=target_keys, device=device)
    # dataset = lmdbDataset('/data/YantiLiu/projects/multispec/datasets/qm9s_raman/qm9s_raman_test.lmdb', target_keys=target_keys, device=device)    
    shuffle = True if mode == 'train' else False
    
    if ddp:
        data_sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=Collator(task),
                              sampler=data_sampler)
        
        if mode == 'train':
            return dataloader, data_sampler
        else: 
            return dataloader
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=Collator(task),
                              num_workers=num_workers, shuffle=shuffle, pin_memory=True)
        return dataloader
