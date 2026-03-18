# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# @Author: Sanvik

import os
import torch
import logging
import sys
import random
# import utils.data_loaders
import utils.helpers
from datetime import datetime
from tqdm import tqdm
from time import time
import numpy as np
import builder
from tensorboardX import SummaryWriter

from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from torch.optim.lr_scheduler import StepLR
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import get_loss
from torch.utils.data import DataLoader
from data.CRN_dataset import CRNShapeNet
from data.ply_dataset import PlyDataset, RealDataset, GeneratedDataset
from data.ScanSalon_dataset import ScanSalonDataset
from utils.loss_utils import get_loss, get_real_loss, get_distill_loss, get_cd, get_ucd
from utils.misc import *
from validate import validate
from collections import OrderedDict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    set_seed(cfg.train.seed)
    torch.backends.cudnn.benchmark = True
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
    cfg.dataset.split = 'train'
    if cfg.dataset.name in ['MatterPort', 'ScanNet', 'KITTI', 'PartNet']:
        train_dataset = RealDataset(cfg)
    elif cfg.dataset.name in ['ModelNet', '3D_FUTURE']:
        train_dataset = GeneratedDataset(cfg)
    elif cfg.dataset.name in ['CRN']:
        train_dataset = CRNShapeNet(cfg)
    elif cfg.dataset.name in ['ScanSalon']:
        train_dataset = ScanSalonDataset(cfg)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=False) # 👈 Ye False hona chahiye

    cfg.dataset.split = 'test'
    if cfg.dataset.name in ['MatterPort', 'ScanNet', 'PartNet','KITTI']:
        val_dataset = RealDataset(cfg)
    elif cfg.dataset.name in ['ModelNet', '3D_FUTURE']:
        val_dataset = GeneratedDataset(cfg)
    elif cfg.dataset.name in ['CRN']:
        val_dataset = CRNShapeNet(cfg)
    elif cfg.dataset.name in ['ScanSalon']:
        val_dataset = ScanSalonDataset(cfg)
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.test.num_workers,
        pin_memory=False) # 🔥 FIX: Changed to False

    output_dir = os.path.join(cfg.train.out_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), '%s')
    cfg.train.checkpoints = output_dir % 'checkpoints'
    cfg.train.logs = output_dir % 'logs'
    log_path = cfg.train.logs + '.txt'
    if not os.path.exists(cfg.train.checkpoints):
        os.makedirs(cfg.train.checkpoints)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.train.logs, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.train.logs, 'test'))

    # Build models
    # model = Point_MAE()
    # model = PCN()
    model = builder.make_model(cfg)
    source_model = builder.make_model(cfg)

    # 1. Surgical Helper: Strips 'module.' and loads weights
    
    def load_cleaned_model(m, path):
        ckpt = torch.load(path, map_location='cpu')
        sd = ckpt['model']
        new_sd = OrderedDict()
        for k, v in sd.items():
            name = k[7:] if k.startswith('module.') else k
            new_sd[name] = v
        
        # Agar model already DataParallel mein wrap ho chuka hai (Resume case)
        target = m.module if hasattr(m, 'module') else m
        target.load_state_dict(new_sd)
        return ckpt

    # Initial Loading for Student and Teacher
    logging.info(f'🔄 Initializing Student & Teacher from: {cfg.train.source_model_path}')
    checkpoint = load_cleaned_model(model, cfg.train.source_model_path)
    source_checkpoint = load_cleaned_model(source_model, cfg.train.source_model_path)
    
    ## 2. Move BOTH models to GPU
    # if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model).cuda()
    #     source_model = torch.nn.DataParallel(source_model).cuda()
    device = torch.device("cuda")
    model = model.to(device)
    source_model = source_model.to(device)
    
    logging.info('✅ Weights loaded and models moved to GPU!')

    optimizer, scheduler = builder.build_opti_sche(model, cfg)
    init_epoch = 0
    best_metrics = float('inf')
    source_ema = EMA(source_model, model)

    # 3. Resume Training Fix (Surgical Strike on 'WEIGHTS' block)
    if 'WEIGHTS' in cfg.train:
        logging.info('Recovering from %s ...' % (cfg.train.model_path))
        recovery_ckpt = load_cleaned_model(model, cfg.train.model_path)
        # ✅ Ye line add kar:
        init_epoch = recovery_ckpt.get('epoch_index', 0) 
        best_metrics = recovery_ckpt.get('best_metrics', float('inf'))
        logging.info(f'✅ Recovery complete. Resuming from Epoch {init_epoch}')

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.train.epochs + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.train()
        source_model.eval()

        total_uhd = 0
        total_ucd = 0
        total_distill = 0
        total_cd_coarse = 0
        total_ucd_coarse = 0
        total_consistency = 0

        batch_end_time = time()
        n_batches = len(train_data_loader)
        with tqdm(train_data_loader) as t:
            for batch_idx, data in enumerate(t):

                data_time.update(time() - batch_end_time)

                if cfg.dataset.name in ['ScanNet', 'MatterPort', 'KITTI', 'PartNet']:
                    partial, index = data
                else:
                    gt, partial, index = data

                bs, _, _ = partial.shape
                partial = partial.float().to(device)
                mask_partial = mask_aug(partial)

                # --- FORWARD ---
                student_out = model(mask_partial.contiguous())
                pcd_pred = student_out[-1]
                coarse_pcd = student_out[0]

                pcd_pred1, pcd_pred2 = pcd_pred[0:bs], pcd_pred[bs:]
                coarse_pcd1, coarse_pcd2 = coarse_pcd[0:bs], coarse_pcd[bs:]

                # --- TEACHER ---
                with torch.no_grad():
                    source_out = source_model(partial.contiguous())
                    coarse_source, source_pred = source_out[0], source_out[-1]

                sample_source = fps_subsample(source_pred, 256)

                # --- LOSSES ---
                loss_ucd_coarse = get_ucd(sample_source, pcd_pred1, sqrt=False) + get_ucd(sample_source, pcd_pred2, sqrt=False)
                loss_cd_coarse = get_cd(coarse_pcd1, coarse_source, sqrt=False) + get_cd(coarse_pcd2, coarse_source, sqrt=False)
                loss_complete = get_ucd(partial, pcd_pred1, sqrt=False) + get_ucd(partial, pcd_pred2, sqrt=False)
                loss_consistency = get_cd(pcd_pred1, pcd_pred2, sqrt=False)

                ucd = loss_complete.item() * 1e4
                ucd_coarse = loss_ucd_coarse.item() * 1e4
                cd_coarse = loss_cd_coarse.item() * 1e4
                consistency = loss_consistency.item() * 1e4

                total_ucd += ucd
                total_ucd_coarse += ucd_coarse
                total_cd_coarse += cd_coarse
                total_consistency += consistency

                loss_total = loss_cd_coarse + loss_complete * 1e2 + loss_ucd_coarse + loss_consistency * 1e2

                # --- BACKWARD ---
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                n_itr = (epoch_idx - 1) * n_batches + batch_idx

                train_writer.add_scalar('Loss/Batch/ucd', ucd, n_itr)
                train_writer.add_scalar('Loss/Batch/ucd_coarse', ucd_coarse, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_coarse', cd_coarse, n_itr)
                train_writer.add_scalar('Loss/Batch/consistency', consistency, n_itr)

                batch_time.update(time() - batch_end_time)
                batch_end_time = time()

                t.set_postfix(loss=['%.4f' % l for l in [ucd, ucd_coarse, cd_coarse, consistency]])

        source_ema.step()

        avg_ucd = total_ucd / n_batches
        avg_ucd_coarse = total_ucd_coarse / n_batches
        avg_cd_coarse = total_cd_coarse / n_batches
        avg_consistency = total_consistency / n_batches

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        epoch_end_time = time()

        train_writer.add_scalar('Loss/Epoch/ucd', avg_ucd, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/ucd_coarse', avg_ucd_coarse, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_coarse', avg_cd_coarse, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/consistency', avg_consistency, epoch_idx)

        print(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            # (epoch_idx, cfg.train.epochs, epoch_end_time - epoch_start_time,['%.4f' % l for l in [avg_ucd, avg_consistency]]))
            (epoch_idx, cfg.train.epochs, epoch_end_time - epoch_start_time,
             ['%.4f' % l for l in [avg_ucd, avg_ucd_coarse, avg_cd_coarse, avg_consistency]]))

        with open(log_path, "a") as file_object:
            # ✅ Sab kuch ek hi line mein bracket ke andar
            file_object.write(f"##########EPOCH {epoch_idx:04d}##########\\n")
            
            # ✅ Is line ko bhi check karle, ek hi line mein honi chahiye
            log_msg = f"Training UCD UCD_Coarse CD_Coarse Consistency: {avg_ucd:.4f} {avg_ucd_coarse:.4f} {avg_cd_coarse:.4f} {avg_consistency:.4f}\\n"
            file_object.write(log_msg)
            
        # Validate the current model
        loss_eval = validate(cfg, epoch_idx, val_dataset, val_data_loader, val_writer, model, source_model, test=False)

        # Save checkpoints
        if epoch_idx % cfg.train.save_freq == 0 or loss_eval < best_metrics:
            file_name = 'ckpt-best.pth' if loss_eval < best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.train.checkpoints, file_name)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': best_metrics,
                'model': model.state_dict()
            }, output_path)
            file_name = 'ckpt-best-source.pth' if loss_eval < best_metrics else 'ckpt-epoch-%03d-source.pth' % epoch_idx
            output_path = os.path.join(cfg.train.checkpoints, file_name)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': best_metrics,
                'model': source_model.state_dict()
            }, output_path)

            print('Saved checkpoint to %s ...' % output_path)
            if loss_eval < best_metrics:
                best_metrics = loss_eval
                with open(log_path, "a") as file_object:
                    file_object.write('Save ckpt-best.pth...................\\n')

    train_writer.close()
    val_writer.close()