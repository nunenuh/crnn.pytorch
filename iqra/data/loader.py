import torch
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader


from .dataset import LMDBDataset, BalanceDatasetConcatenator
import torchvision.transforms as VT
from .. import transforms as NT


def train_loader(path, batch_size=32, shuffle=True, num_workers=8,
                 img_size=(32, 100), usage_ratio=(0.5, 0.5), is_sensitive=True,
                 character="0123456789abcdefghijklmnopqrstuvwxyz"):

    trn_transform = VT.Compose([
        NT.ResizeRatioWithRightPad(size=img_size),
        VT.ToTensor(),
        VT.Normalize(mean=(0.5), std=(0.5))
    ])
    subdir_names = ['ST', 'MJ']
    bdc = BalanceDatasetConcatenator(path, dataset_class=LMDBDataset,
                                     transform=trn_transform,
                                     subdir=subdir_names, usage_ratio=usage_ratio,
                                     im_size=img_size, is_sensitive=is_sensitive,
                                     character=character)
    dset = bdc.get_dataset()

    dloader = DataLoader(dset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         pin_memory=True)

    return dloader, dset


def valid_loader(path, batch_size=32, shuffle=True, num_workers=8,
                 img_size=(32, 100), is_sensitive=True,
                 character="0123456789abcdefghijklmnopqrstuvwxyz"):

    val_transform = VT.Compose([
        NT.ResizeRatioWithRightPad(size=img_size),
        VT.ToTensor(),
        VT.Normalize(mean=(0.5), std=(0.5))
    ])

    bdc = BalanceDatasetConcatenator(path, dataset_class=LMDBDataset,
                                     transform=val_transform,
                                     im_size=img_size, is_sensitive=is_sensitive,
                                     character=character)
    dset = bdc.get_dataset()

    dloader = DataLoader(dset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         pin_memory=True)
    return dloader, dset


def test_loader(path, batch_size=32, shuffle=False, num_workers=8,
                 img_size=(32, 100), is_sensitive=True,
                 character="0123456789abcdefghijklmnopqrstuvwxyz"):
    
    val_transform = VT.Compose([
        NT.ResizeRatioWithRightPad(size=img_size),
        VT.ToTensor(),
        VT.Normalize(mean=(0.5), std=(0.5))
    ])
    
#     subdir_names = ['CUTE80','IC03_860','IC03_867',
#                    'IC13_857','IC13_1015','IC15_1811',
#                    'IC15_2077','IIIT5k_3000','SVT',
#                     'SVTP']
#     usage_ratio = [1.0 for i in range(len(subdir_names))]
    

    bdc = BalanceDatasetConcatenator(path, dataset_class=LMDBDataset,
                                     transform=val_transform,
                                     im_size=img_size, is_sensitive=is_sensitive,
                                     character=character)
    dset = bdc.get_dataset()

    dloader = DataLoader(dset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         pin_memory=True)
    return dloader, dset
