#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Sanvik 

import sys
import os

# 🔥 --- PATH HIJACK FOR LOCAL CORE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
H100_CORE = os.path.join(BASE_DIR, 'h100_core')
if os.path.exists(H100_CORE):
    sys.path.insert(0, H100_CORE)
    sys.path = [p for p in sys.path if 'site-packages/pointnet2_ops' not in p]
# ---------------------------------

import logging
import argparse
import numpy as np
import torch
from utils import yaml_reader
from pprint import pprint
from train import train
from test import test

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--config', type=str, default='./configs/snow.yaml', help='Configuration File')
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    #parser.add_argument('--test', dest='test', help='Test neural network in real datasets', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args_from_command_line()
    cfg = yaml_reader.read_yaml(args.config)

    set_seed(cfg.train.seed)
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
    torch.backends.cudnn.benchmark = True

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg.train.device)
    
    if args.test:
        test(cfg)
    else:
        train(cfg)