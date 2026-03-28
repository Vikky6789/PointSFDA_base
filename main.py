#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Sanvik 

import sys
import os

# 🔥 --- THE ULTIMATE PATH HIJACK ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Har compiled module ka exact address Python ko de do
sys.path.insert(0, os.path.join(BASE_DIR, 'pointnet2_ops_lib'))
sys.path.insert(0, os.path.join(BASE_DIR, 'extensions', 'expansion_penalty'))
sys.path.insert(0, os.path.join(BASE_DIR, 'extensions', 'chamfer_dist'))
sys.path.insert(0, os.path.join(BASE_DIR, 'Chamfer3D'))
sys.path.insert(0, BASE_DIR)
# ----------------------------------

import logging
import argparse
import numpy as np
import torch
from utils import yaml_reader
from train import train
from test import test

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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
    parser.add_argument('--use_pointmac', action='store_true', help='Enable PointMAC MAML Training & TTA')
    # argparse setup ke andar:
    parser.add_argument('--use_gan', action='store_true', help='Enable Coarse Adversarial Alignment (GAN)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args_from_command_line()
    cfg = yaml_reader.read_yaml(args.config)

    cfg.use_pointmac = args.use_pointmac
    if 'model' in cfg: cfg.model.use_pointmac = args.use_pointmac
    
    # Jahan config load ho rahi hai:
    cfg.use_gan = args.use_gan
    
    set_seed(cfg.train.seed)
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
    torch.backends.cudnn.benchmark = True

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg.train.device)
    
    if args.test:
        test(cfg)
    else:
        train(cfg)