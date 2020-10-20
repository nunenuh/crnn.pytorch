import os
import sys
sys.path.append(os.getcwd())

from pathlib import Path


import numpy as np
import random

import torch

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import argparse

import string
from iqra.utils import AttnLabelConverter
from iqra.data import loader
from iqra.models import OCRNet
from iqra.trainer.task import TaskOCR

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.metrics import Accuracy



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='crnn.pytorch trainer cli apps')
    parser.add_argument('--resume', default=None, type=str, help='Choose pth file to resume training')


    parser.add_argument('--manual_seed', type=int, default=1111, help='for random seed setting')

    parser.add_argument('--max_epoch', required=True, default=None,
                        type=int, help='How many epoch to run training')
    parser.add_argument('--lr', '--learning-rate', default=1, type=float,
                        help='choose learning rate for optimizer, default value is 0.01')
    parser.add_argument('--beta1',  default=0.9, type=float,
                        help='choose beta1 for optimizer, default value is 0.9')
    parser.add_argument('--beta2',  default=0.95, type=float,
                        help='choose beta2 for optimizer, default value is 0.999')
    
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    
    parser.add_argument('--grad_clip', default=5.0, type=float,
                        help='choose gradient clip value for backward prop, default value is 5.0')

    parser.add_argument('--batch_size', default=32, type=int,
                        help='choose batch size for data loader, default value is 32')
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='choose to shuffle data or not, default value is True')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='how many workers to load for running dataset')

    parser.add_argument('--trainset_path', required=True, type=str,
                        help='path to synthtext dataset')
    parser.add_argument('--validset_path', required=True, type=str,
                        help='path to synthtext dataset')
    parser.add_argument('--image_size', default='100x32', type=str,
                        help='width and height of the image, default value is 100x32')
    parser.add_argument('--usage_ratio', default='0.5,0.5', type=str,
			help='training data usage ratio default is (0.5, 0.5)')
    
    parser.add_argument('--batch_max_length', default=25, type=int,
                        help='choose batch size for data loader, default value is 32')
    
    
    parser.add_argument('--character', type=str,  default='0123456789abcdefghijklmnopqrstuvwxyz',
                         help='character label')
    parser.add_argument('--sensitive', type=bool, default=True, help='for sensitive character mode')
    
    
    parser.add_argument('--in_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--out_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    parser.add_argument('--resnet_version', type=int, default=18, help='choose pretrained resnet version')

    

    parser.add_argument('--num_gpus', default=1, type=int,
                        help='fill with zero to use cpu or fill with number 2 to use multigpu')
    parser.add_argument('--log_freq', default=10, type=int,
                        help='show log every value, default value is 10')
    
    
    parser.add_argument('--max_steps', default=30000, type=int,
                        help='max iteration step, default value is 30000')
    
    parser.add_argument('--valcheck_interval', default=2000, type=int,
                        help='validation check interval in step, default value is 2000')

    parser.add_argument('--checkpoint_dir', default='saved_checkpoints/', type=str,
                        help='checkpoint directory for saving progress')
    parser.add_argument('--logs_dir', default='logs/', type=str,
                        help='directory logs for tensorboard callback')

    args = parser.parse_args()

    w, h = args.image_size.split('x')
    w, h = int(w), int(h)

    MANUAL_SEED = args.manual_seed
    BENCHMARK = True
    DETERMINISTIC = True
    

    MAX_EPOCH = args.max_epoch
    MAX_STEPS = args.max_steps
    
    
    LRATE = args.lr
    BETA1 = args.beta1
    BETA2 = args.beta2
    RHO = args.rho
    EPS = args.eps
    GRAD_CLIP = args.grad_clip
    
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    SHUFFLE = args.shuffle
    IMG_SIZE = (h, w)
    USAGE_RATIO = list(map(float, args.usage_ratio.split(',')))
    # print(USAGE_RATIO)
    
    BATCH_MAX_LENGTH = args.batch_max_length
    SENSITIVE = args.sensitive
    if SENSITIVE:
        CHARACTER = string.printable[:-6]
    else:
        CHARACTER = args.character
        
    TRAINSET_PATH = args.trainset_path
    VALIDSET_PATH = args.validset_path
    
    
    IN_CHANNEL = args.in_channel
    OUT_CHANNEL = args.out_channel
    HIDDEN_SIZE = args.hidden_size
    RESNET_VERSION = args.resnet_version
    
    
    NUM_GPUS = args.num_gpus
    
    
    SAVED_CHECKPOINT_PATH = args.checkpoint_dir
    SAVED_LOGS_PATH = args.logs_dir
    LOG_FREQ = args.log_freq
    VALCHECK_INTERVAL = args.valcheck_interval
    
    
    CHECKPOINT_RESUME = False
    CHECKPOINT_PATH = None
    
    WEIGHT_RESUME = False
    WEIGHT_PATH = None
    
    
    if args.resume:
        fpath = Path(args.resume)
        if fpath.is_file():
            if fpath.suffix == 'ckpt':
                # it means checkpoint of pytorch lightning 
                CHECKPOINT_RESUME = True
                CHECKPOINT_PATH = str(fpath)
            elif fpath.suffix == 'pth':
                # it means pytorch file original from model
                WEIGHT_RESUME = True
                WEIGHT_PATH = str(fpath)       
            else:
                raise NotImplemented(f'File with {fpath.suffix} is not implemented! ' 
                                     f'make sure you load valid file with ckpt or pth extension!')
        else:
            raise IOError(f'Path that you specified is not valid pytorch or pytorch-lighning path!')
    
    converter = AttnLabelConverter(CHARACTER)
    NUM_CLASS = len(converter.character)
    
    
    trainloader, trainset = loader.train_loader(TRAINSET_PATH, batch_size=BATCH_SIZE, 
                                      shuffle=SHUFFLE, num_workers=NUM_WORKERS,
                                      img_size=IMG_SIZE, usage_ratio=USAGE_RATIO,
				                      is_sensitive=SENSITIVE, character=CHARACTER)
    
    validloader, validset = loader.valid_loader(VALIDSET_PATH, batch_size=BATCH_SIZE,
                                      shuffle=False, num_workers=NUM_WORKERS,
                                      img_size=IMG_SIZE, is_sensitive=SENSITIVE,
                                      character=CHARACTER)    
    
    # Model Preparation
    if WEIGHT_RESUME:
        model = OCRNet(num_class=NUM_CLASS, in_feat=IN_CHANNEL, hidden_size=HIDDEN_SIZE, im_size=IMG_SIZE,
			resnet_version=RESNET_VERSION, pretrained_feature=True, freeze_feature=True)
        weights = torch.load(WEIGHT_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(weights)
        
        transformer_weight = torch.load('weights/spatial_transformer.pth', map_location=torch.device('cpu'))
        model.encoder.transformer.load_state_dict(transformer_weight)
    else:
        model = OCRNet(num_class=NUM_CLASS, in_feat=IN_CHANNEL, hidden_size=HIDDEN_SIZE, im_size=IMG_SIZE,
			resnet_version=RESNET_VERSION, pretrained_feature=True, freeze_feature=True)

        transformer_weight = torch.load('weights/spatial_transformer.pth', map_location=torch.device('cpu'))
        model.encoder.transformer.load_state_dict(transformer_weight)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adadelta(model.parameters(), lr=LRATE, rho=RHO, eps=EPS)
    task = TaskOCR(model, optimizer, criterion, converter)
    
    # DEFAULTS used by the Trainer
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=SAVED_CHECKPOINT_PATH,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix='ocrnet'
    )
    

    tb_logger = pl_loggers.TensorBoardLogger(SAVED_LOGS_PATH)
    
    DISTRIBUTED_BACKEND = None
    if NUM_GPUS > 1:
        DISTRIBUTED_BACKEND = 'ddp'
     
    # seed befire train
    pl.trainer.seed_everything(MANUAL_SEED)
    
    trainer = pl.Trainer(
        weights_summary="top",
        max_epochs=MAX_EPOCH,
        max_steps=MAX_STEPS,
        val_check_interval=VALCHECK_INTERVAL,
        gpus=NUM_GPUS,
        distributed_backend=DISTRIBUTED_BACKEND,
        log_every_n_steps=LOG_FREQ,
        deterministic=DETERMINISTIC,
        benchmark=BENCHMARK,
        logger=tb_logger, 
        checkpoint_callback=checkpoint_callback, 
        resume_from_checkpoint=CHECKPOINT_PATH
    )


    
    trainer.fit(task, trainloader, validloader)
