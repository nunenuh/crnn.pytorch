import os
import sys
import re
import six
import math
import lmdb
from pathlib import Path
from typing import *


import numpy as np
from natsort import natsorted
from PIL import Image


import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms

class LMDBDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, im_size=(32,100), is_sensitive=True, **kwargs):
        """[summary]

        Args:
            root ([type]): [description]
            im_size (tuple, optional): [description]. Defaults to (32,100).
            is_sensitive (bool, optional): [description]. Defaults to True.
        """
        self._init_default_attrs(**kwargs)
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.im_size = im_size
        self.im_height, self.im_width = self.im_size
        self.is_sensitive = is_sensitive
        
        self._build_lmdb_env()
        
    def _init_default_attrs(self, **kwargs):
        self.data_filtering_off: bool = kwargs.get('data_filtering_off', False)
        self.batch_max_length: int = kwargs.get('batch_max_length', 25)
        self.is_rgb: bool = kwargs.get('is_rgb', False)
        self.character = kwargs.get('character', '0123456789abcdefghijklmnopqrstuvwxyz')
        
        
    def _build_lmdb_env(self):
        self.env = lmdb.open(self.root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print(f'cannot create lmdb from {self.root}')
            sys.exit(0)
            
        with self.env.begin(write=False) as txn:
            nsamples = int(txn.get('num-samples'.encode()))
            self.nsamples = nsamples
            
            if self.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nsamples)]
            else:
               
                self.filtered_index_list = []
                for index in range(self.nsamples):
                    index = index + 1 # lmdb starts with 1
                    label_key = f'label-{index:09d}'.encode()
                    label = txn.get(label_key).decode('utf-8')
                    
                    if len(label) > self.batch_max_length:
                         # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue
                    
                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)
                
                self.nsamples = len(self.filtered_index_list)
                
    def __len__(self):
        return self.nsamples
    
    def _get_lmdb_data(self, index):
        with self.env.begin(write=False) as txn:
            label_key = f'label-{index:09d}'.encode()
            label = txn.get(label_key).decode('utf-8')
            img_key = f'image-{index:09d}'.encode()
            imgbuf = txn.get(img_key)
            
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                if self.is_rgb:
                    img = Image.open(buf).convert('RGB')
                else:
                    img = Image.open(buf).convert('L')
            except IOError:
                print(f'Corrupted image for {index}')
                 # make dummy image and dummy label for corrupted image.
                if self.is_rgb:
                    img = Image.new('RGB', (self.im_width, self.im_height))
                else:
                    img = Image.new('L', (self.im_width, self.im_height))
                label = '[dummy_label]'
            
            if not self.is_sensitive:
                label = label.lower()
                
            
            out_of_char = f'[^{self.character}]'
            label = re.sub(out_of_char, '', label)
            
        return img, label
        
    def __getitem__(self, index):
        assert index <= len(self), 'Index range error!'
        index = self.filtered_index_list[index]
        img, label = self._get_lmdb_data(index)
        
        if self.transform:
            img = self.transform(img)
        
        if self.target_transform:
            label = self.target_transform(label)
            
        return (img, label)


class BalanceDatasetConcatenator(object):
    def __init__(self, root, subdir: Tuple[str] = ('/',), usage_ratio: Tuple =(1,), dataset_class=LMDBDataset, **kwargs):
        assert len(subdir) == len(usage_ratio), "subdir must have same item number with ratio!"
        self.root = root
        self.subdir = subdir
        self.usage_ratio = usage_ratio
        self.DatasetClass = dataset_class
        self.dataset_params = kwargs
        self.dataset_list = []
        self.concat_dataset = None
        self.total_length = 0
        
        self._create_and_fill_base_datamap()

    def _walktree_dir(self, path):
        for dirpath, dirname, _f in os.walk(path+'/'):
            if not dirname:
                yield Path(dirpath)
                
    def _create_base_datamap(self):
        """
        create dictionary for concat and splitting latter with this sample data
        {
            'MJ': {
                'ratio': 0.5,
                'dirpath': [],
                'total_length': 0
            }
        }
        """
        
        dct_data = {}
        list_dict = [{sd: {'ratio': ur, 'dirpath':[], 'total_length':0}} for sd, ur in zip(self.subdir, self.usage_ratio)]
        for lsd in list_dict: dct_data.update(lsd)
        return dct_data

    def _create_and_fill_base_datamap(self):
        self.base_datamap = self._create_base_datamap()
        for dirpath in self._walktree_dir(self.root):
            select_flag = False
            for key in self.subdir:
                if key in str(dirpath):
                    select_flag = True
                    break
            
            if select_flag:
                self.base_datamap[key]['dirpath'].append(str(dirpath))
          

    def _build_balanced_dataset(self):
        concat_list = []
        tlength = 0
        for key, data in self.base_datamap.items():
            dataset_list = []
            ratio = self.base_datamap[key]['ratio']
            for subdir_path in self.base_datamap[key]['dirpath']:
                dataset = self.DatasetClass(root=str(subdir_path), **self.dataset_params)
                dataset_tlen = len(dataset)
                dataset_trn_len = int(dataset_tlen * ratio)
                dataset_val_len = dataset_tlen - dataset_trn_len
                indices = [dataset_trn_len, dataset_val_len]
                
                trainset, validset = random_split(dataset, lengths=indices)
                dataset_list.append(trainset)
                tlength += dataset_trn_len 
            
            concatset = ConcatDataset(dataset_list)
            concat_list.append(concatset)
            self.base_datamap[key]['total_length'] = tlength
            
            self.total_length += tlength
            
        self.balanced_dataset = ConcatDataset(concat_list)
        
        
        
    
    def get_dataset(self):
        self._build_balanced_dataset()
        return self.balanced_dataset
        

class RawDataset(Dataset):
    
    def __init__(self, root, is_rgb=False, im_size=(32, 100)):
        self.root = root
        self.is_rgb = is_rgb
        self.im_size = im_size
        self.im_height, self.im_width = self.im_size
        
        self.image_path_list = self._create_image_path_list(root)
        self.nsamples = len(self.image_path_list)
        
    def _create_image_path_list(self, root):
        image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    image_path_list.append(os.path.join(dirpath, name))
        image_path_list = natsorted(image_path_list)
        return image_path_list
    
    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        try:
            if self.is_rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')
            else:
                img = Image.open(self.image_path_list[index]).convert('L')
        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.is_rgb:
                img = Image.new('RGB', (self.im_width, self.im_height))
            else:
                img = Image.new('L', (self.im_width, self.im_height))

        return (img, self.image_path_list[index])
            


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
