# # -*- coding: utf-8 -*-
# # -*- coding: utf-8 -*-
# # @Author: Sanvik

# import os
# import torch
# import logging
# import sys
# import random
# # import utils.data_loaders
# import utils.helpers
# from datetime import datetime
# from tqdm import tqdm
# from time import time
# import numpy as np
# import builder
# from tensorboardX import SummaryWriter

# from utils.average_meter import AverageMeter
# from utils.metrics import Metrics
# from torch.optim.lr_scheduler import StepLR
# from utils.schedular import GradualWarmupScheduler
# from utils.loss_utils import get_loss
# from torch.utils.data import DataLoader
# from data.CRN_dataset import CRNShapeNet
# from data.ply_dataset import PlyDataset, RealDataset, GeneratedDataset
# from data.ScanSalon_dataset import ScanSalonDataset
# from utils.loss_utils import get_loss, get_real_loss, get_distill_loss, get_cd, get_ucd
# from utils.misc import *
# from validate import validate
# from collections import OrderedDict

# # === 🚀 NEW IMPORT FOR ADVERSARIAL ALIGNMENT ===
# from adversarial_alignment.discriminator import CoarsePointDiscriminator
# import torch.nn as nn
# # ===============================================

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True

# def train(cfg):
#     # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
#     set_seed(cfg.train.seed)
#     torch.backends.cudnn.benchmark = True
#     logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
#     cfg.dataset.split = 'train'
#     if cfg.dataset.name in ['MatterPort', 'ScanNet', 'KITTI', 'PartNet']:
#         train_dataset = RealDataset(cfg)
#     elif cfg.dataset.name in ['ModelNet', '3D_FUTURE']:
#         train_dataset = GeneratedDataset(cfg)
#     elif cfg.dataset.name in ['CRN']:
#         train_dataset = CRNShapeNet(cfg)
#     elif cfg.dataset.name in ['ScanSalon']:
#         train_dataset = ScanSalonDataset(cfg)

#     train_data_loader = DataLoader(
#         train_dataset,
#         batch_size=cfg.train.batch_size,
#         shuffle=True,
#         num_workers=cfg.train.num_workers,
#         pin_memory=True) # 👈 Ye False hona chahiye

#     cfg.dataset.split = 'test'
#     if cfg.dataset.name in ['MatterPort', 'ScanNet', 'PartNet','KITTI']:
#         val_dataset = RealDataset(cfg)
#     elif cfg.dataset.name in ['ModelNet', '3D_FUTURE']:
#         val_dataset = GeneratedDataset(cfg)
#     elif cfg.dataset.name in ['CRN']:
#         val_dataset = CRNShapeNet(cfg)
#     elif cfg.dataset.name in ['ScanSalon']:
#         val_dataset = ScanSalonDataset(cfg)
#     val_data_loader = DataLoader(
#         val_dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=cfg.test.num_workers,
#         pin_memory=True) # 🔥 FIX: Changed to False

#     output_dir = os.path.join(cfg.train.out_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), '%s')
#     cfg.train.checkpoints = output_dir % 'checkpoints'
#     cfg.train.logs = output_dir % 'logs'
#     log_path = cfg.train.logs + '.txt'
#     if not os.path.exists(cfg.train.checkpoints):
#         os.makedirs(cfg.train.checkpoints)

#     # Create tensorboard writers
#     train_writer = SummaryWriter(os.path.join(cfg.train.logs, 'train'))
#     val_writer = SummaryWriter(os.path.join(cfg.train.logs, 'test'))

#     # Build models
#     # model = Point_MAE()
#     # model = PCN()
#     model = builder.make_model(cfg)
#     source_model = builder.make_model(cfg)

#     # 1. Surgical Helper: Strips 'module.' and loads weights
    
#     def load_cleaned_model(m, path):
#         ckpt = torch.load(path, map_location='cpu')
#         sd = ckpt['model']
#         new_sd = OrderedDict()
#         for k, v in sd.items():
#             name = k[7:] if k.startswith('module.') else k
#             new_sd[name] = v
        
#         # Agar model already DataParallel mein wrap ho chuka hai (Resume case)
#         target = m.module if hasattr(m, 'module') else m
#         target.load_state_dict(new_sd, strict=False)
#         return ckpt

#     # Initial Loading for Student and Teacher
#     logging.info(f'🔄 Initializing Student & Teacher from: {cfg.train.source_model_path}')
#     checkpoint = load_cleaned_model(model, cfg.train.source_model_path)
#     source_checkpoint = load_cleaned_model(source_model, cfg.train.source_model_path)
    
#     ## 2. Move BOTH models to GPU
#     # if torch.cuda.is_available():
#     #     model = torch.nn.DataParallel(model).cuda()
#     #     source_model = torch.nn.DataParallel(source_model).cuda()
#     device = torch.device("cuda")
#     model = model.to(device)
#     source_model = source_model.to(device)
#     scaler = torch.amp.GradScaler('cuda') # H100 ke liye updated syntax
    
#     logging.info('✅ Weights loaded and models moved to GPU!')

#     optimizer, scheduler = builder.build_opti_sche(model, cfg)
    
#     # === ⚖️ Coarse GAN Setup ===
#     discriminator = None # Default value taaki aage crash na ho
#     alpha_gan = 0.0      # Default weight 0
    
#     if getattr(cfg, 'use_gan', False):
#         logging.info("⚖️ Initializing Coarse Adversarial Discriminator...")
#         discriminator = CoarsePointDiscriminator().to(device)
#         optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
#         criterion_gan = nn.MSELoss().to(device)
#         alpha_gan = 0.05
#     # ==========================
    
#     # 👇 YE BLOCK ADD KAR (PointMAC Lambda Setup):
#     if getattr(cfg, 'use_pointmac', False):
#         logging.info("🧠 Initializing PointMAC Meta-Learning Lambdas...")
#         param_device = next(model.parameters()).device
#         lambda_smr = torch.tensor(0.0, device=param_device, requires_grad=True)
#         lambda_ad = torch.tensor(0.0, device=param_device, requires_grad=True)
#         lambda_optimizer = torch.optim.Adam([lambda_smr, lambda_ad], lr=1e-3)
#         meta_weight = 1.0
#     # 👆 ==========================================
    
#     init_epoch = 0
#     best_metrics = float('inf')
#     source_ema = EMA(source_model, model)

#     # 3. Resume Training Fix (Surgical Strike on 'WEIGHTS' block)
#     if 'WEIGHTS' in cfg.train:
#         logging.info('Recovering from %s ...' % (cfg.train.model_path))
#         recovery_ckpt = load_cleaned_model(model, cfg.train.model_path)
#         # ✅ Ye line add kar:
#         init_epoch = recovery_ckpt.get('epoch_index', 0) 
#         best_metrics = recovery_ckpt.get('best_metrics', float('inf'))
#         logging.info(f'✅ Recovery complete. Resuming from Epoch {init_epoch}')

#     # Training/Testing the network
#     for epoch_idx in range(init_epoch + 1, cfg.train.epochs + 1):
#         epoch_start_time = time()

#         batch_time = AverageMeter()
#         data_time = AverageMeter()

#         model.train()
#         source_model.eval()
#         if discriminator is not None:
#             discriminator.train()

#         total_uhd = 0
#         total_ucd = 0
#         total_distill = 0
#         total_cd_coarse = 0
#         total_ucd_coarse = 0
#         total_consistency = 0

#         batch_end_time = time()
#         n_batches = len(train_data_loader)
#         with tqdm(train_data_loader) as t:
#             for batch_idx, data in enumerate(t):
#                 data_time.update(time() - batch_end_time)
#                 if cfg.dataset.name in ['ScanNet', 'MatterPort', 'KITTI', 'PartNet']: # Added PartNet
#                     partial, index = data
#                 elif cfg.dataset.name in ['ModelNet', '3D_FUTURE', 'CRN', 'ScanSalon']:
#                     gt, partial, index = data
                
#                 ### WITHOUT H100
                    
#                 # bs, _, _ = partial.shape
#                 # partial = partial.float().cuda()  # [16, 2048, 3]
#                 # mask_partial = mask_aug(partial)


#                 # student_out = model(mask_partial.contiguous())
#                 # pcd_pred = student_out[-1] #[32, 2048, 3]
#                 # coarse_pcd = student_out[0]
#                 # # b = int(bs/2)
                
#                 # pcd_pred1 = pcd_pred[0:bs, :, :]
#                 # pcd_pred2 = pcd_pred[bs:, :, :]
                
#                 # assert pcd_pred1.shape == pcd_pred2.shape
#                 # coarse_pcd1 = coarse_pcd[0:bs, :, :]  ###
#                 # coarse_pcd2 = coarse_pcd[bs:, :, :]  ###
        
#                 # assert coarse_pcd1.shape == coarse_pcd2.shape

#                 # with torch.no_grad():

#                 #     source_out = source_model(partial.contiguous())
#                 #     coarse_source = source_out[0]  ######
#                 #     source_pred = source_out[-1]

#                 # sample_source = fps_subsample(source_pred, 256)  #####
#                 # loss_ucd_coarse = get_ucd(sample_source, pcd_pred1, sqrt=False) + get_ucd(sample_source, pcd_pred2, sqrt=False)

#                 # ucd_coarse = loss_ucd_coarse.item() * 1e4
#                 # total_ucd_coarse += ucd_coarse

#                 # assert coarse_pcd1.shape[1] == coarse_pcd2.shape[1] == coarse_source.shape[1]
#                 # loss_cd_coarse = get_cd(coarse_pcd1, coarse_source, sqrt=False) + get_cd(coarse_pcd2, coarse_source,sqrt=False)
#                 # cd_coarse = loss_cd_coarse.item() * 1e4
#                 # total_cd_coarse += cd_coarse

#                 # loss_complete = get_ucd( partial,pcd_pred1, sqrt=False) + get_ucd( partial,pcd_pred2, sqrt=False)  # L2UCD UHD
#                 # ucd = loss_complete.item() * 1e4
#                 # total_ucd += ucd

#                 # loss_consistency = get_cd(pcd_pred1, pcd_pred2, sqrt=False)
#                 # consistency = loss_consistency.item() * 1e4
#                 # total_consistency += consistency

#                 # loss_total = loss_cd_coarse + loss_complete * 1e2 + loss_ucd_coarse + loss_consistency * 1e2

#                 # optimizer.zero_grad()
#                 # loss_total.backward()
#                 # optimizer.step()
                
                
#                 # Purana logic hata kar ye daal:
#                 bs, _, _ = partial.shape
#                 partial = partial.float().to(device)
#                 mask_partial = mask_aug(partial)

#                 # 🔥 H100 Power Mode Start
#                 with torch.amp.autocast("cuda", enabled=False):
#                     mask_partial = mask_partial.float()
                    
#                     # 👇 POINTMAC MODULAR FORWARD PASS
#                     if getattr(cfg, 'use_pointmac', False):
#                         student_out, aux_outputs = model(mask_partial.contiguous(), return_aux=True)
#                     else:
#                         student_out = model(mask_partial.contiguous())
#                     # 👆 ==========================================
                    
#                     pcd_pred = student_out[-1]
#                     coarse_pcd = student_out[0]
                    
#                     pcd_pred1, pcd_pred2 = pcd_pred[0:bs], pcd_pred[bs:]
#                     coarse_pcd1, coarse_pcd2 = coarse_pcd[0:bs], coarse_pcd[bs:]

#                     with torch.no_grad():
#                         source_out = source_model(partial.contiguous())
#                         coarse_source, source_pred = source_out[0], source_out[-1]

#                     sample_source = fps_subsample(source_pred, 256)
                    
#                     # Losses (Isi block ke andar rahenge)
#                     loss_ucd_coarse = get_ucd(sample_source, pcd_pred1, sqrt=False) + get_ucd(sample_source, pcd_pred2, sqrt=False)
#                     loss_cd_coarse = get_cd(coarse_pcd1, coarse_source, sqrt=False) + get_cd(coarse_pcd2, coarse_source, sqrt=False)
#                     loss_complete = get_ucd(partial, pcd_pred1, sqrt=False) + get_ucd(partial, pcd_pred2, sqrt=False)
#                     loss_consistency = get_cd(pcd_pred1, pcd_pred2, sqrt=False)

#                     # --- STATS EXTRACTION ---
#                     ucd = loss_complete.item() * 1e4
#                     ucd_coarse = loss_ucd_coarse.item() * 1e4
#                     cd_coarse = loss_cd_coarse.item() * 1e4
#                     consistency = loss_consistency.item() * 1e4

#                     # Epoch totals update
#                     total_ucd += ucd
#                     total_ucd_coarse += ucd_coarse
#                     total_cd_coarse += cd_coarse
#                     total_consistency += consistency
                    
#                     # Base Loss (Original)
#                     loss_main = loss_cd_coarse + loss_complete * 1e2 + loss_ucd_coarse + loss_consistency * 1e2
#                     loss_total = loss_main

#                     # ---------------------------------------------------------
#                     # 2. POINTMAC AUXILIARY LOSS CALCULATION & LAMBDA BALANCING
#                     if getattr(cfg, 'use_pointmac', False):
#                         mae_rec = aux_outputs['mae_rec']                # [2*bs, N, 3]
#                         denoise_offset = aux_outputs['denoise_offset']  # [2*bs, M, 3]
                        
#                         # Split just like your original predictions
#                         mae_rec1, mae_rec2 = mae_rec[0:bs], mae_rec[bs:]
                        
#                         # Aux1: Masking Loss (Compare reconstructed mask with original partial)
#                         loss_aux1 = get_cd(mae_rec1, partial, sqrt=False) * 1e2 + get_cd(mae_rec2, partial, sqrt=False) * 1e2
                        
#                         # Aux2: Denoising Offset Loss (Minimize offset magnitude)
#                         loss_aux2 = torch.mean(denoise_offset ** 2) * 1e2 

#                         # Meta-learned weighting λ (Adaptive Balancer)
#                         alpha_tilde = torch.log(1 + lambda_smr ** 2)
#                         beta_tilde = torch.log(1 + lambda_ad ** 2)
#                         w_smr = torch.exp(alpha_tilde) / (torch.exp(alpha_tilde) + torch.exp(beta_tilde))
#                         w_ad = 1.0 - w_smr

#                         # Final Loss Combining Main and Aux
#                         loss_aux_total = w_smr * loss_aux1 + w_ad * loss_aux2
#                         loss_total = loss_main + meta_weight * loss_aux_total
#                     # ---------------------------------------------------------

#                     # =========================================================
#                     # 🥊 NEW GAN CODE: ADVERSARIAL ALIGNMENT (DISCRIMINATOR)
#                     # =========================================================
#                     loss_G_adv = torch.tensor(0.0) # Default value taaki aage logger crash na kare
                    
#                     if getattr(cfg, 'use_gan', False):
#                         coarse_real = coarse_source.detach()
#                         coarse_fake = coarse_pcd1
                        
#                         valid_labels = torch.ones((bs, 1), dtype=torch.float32, device=device)
#                         fake_labels = torch.zeros((bs, 1), dtype=torch.float32, device=device)

#                         # --- Step A: Train Discriminator ---
#                         optimizer_D.zero_grad()
#                         loss_D_real = criterion_gan(discriminator(coarse_real), valid_labels)
#                         loss_D_fake = criterion_gan(discriminator(coarse_fake.detach()), fake_labels)

#                         loss_D = 0.5 * (loss_D_real + loss_D_fake)
#                         scaler.scale(loss_D).backward()
#                         scaler.step(optimizer_D)
                        
#                         # --- Step B: Generator (Target Model) Adversarial Loss ---
#                         loss_G_adv = criterion_gan(discriminator(coarse_fake), valid_labels)
#                         loss_total = loss_total + (alpha_gan * loss_G_adv)
#                     # =========================================================

#                 # 🔥 Optimized Backward
#                 optimizer.zero_grad()
#                 if getattr(cfg, 'use_pointmac', False):
#                     lambda_optimizer.zero_grad() # Clear lambda gradients
                    
#                 scaler.scale(loss_total).backward() 
    
#                 scaler.step(optimizer)
#                 if getattr(cfg, 'use_pointmac', False):
#                     scaler.step(lambda_optimizer) # Update lambdas
                
#                 scaler.update()
                
#                 n_itr = (epoch_idx - 1) * n_batches + batch_idx

#                 train_writer.add_scalar('Loss/Batch/ucd', ucd, n_itr)
#                 train_writer.add_scalar('Loss/Batch/ucd_coarse', ucd_coarse, n_itr)
#                 train_writer.add_scalar('Loss/Batch/cd_coarse', cd_coarse, n_itr)
#                 train_writer.add_scalar('Loss/Batch/consistency', consistency, n_itr)

#                 batch_time.update(time() - batch_end_time)
#                 batch_end_time = time()
                
#                 t.set_description(
#                     '[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.train.epochs, batch_idx + 1, n_batches))
#                 t.set_postfix(loss='%s' % ['%.4f' % l for l in [ucd, ucd_coarse, cd_coarse, consistency]])
#                 # 🔥 TERA CUSTOM STEP-BY-STEP LOGGER 🔥
#                 # Har 1 batch ke baad terminal pe ek detail print karega
#                 if batch_idx % 20 == 0:
#                     logging.info(f"👉 [Epoch {epoch_idx}] Batch {batch_idx}/{n_batches} processed! | UCD: {ucd:.4f} | CD Coarse: {cd_coarse:.4f} | GAN Loss: {loss_G_adv.item():.4f}")
#                 # t.set_postfix(loss='%s' % ['%.4f' % l for l in [ucd,  consistency]])
#                 if cfg.scheduler.type == 'GradualWarmup':
#                     if n_itr < cfg.scheduler.kwargs_2.total_epoch:
#                         scheduler.step()

#         source_ema.step()

#         avg_ucd = total_ucd / n_batches
#         avg_ucd_coarse = total_ucd_coarse / n_batches
#         avg_cd_coarse = total_cd_coarse / n_batches
#         avg_consistency = total_consistency / n_batches

#         if isinstance(scheduler, list):
#             for item in scheduler:
#                 item.step()
#         else:
#             scheduler.step()
#         print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
#         epoch_end_time = time()

#         train_writer.add_scalar('Loss/Epoch/ucd', avg_ucd, epoch_idx)
#         train_writer.add_scalar('Loss/Epoch/ucd_coarse', avg_ucd_coarse, epoch_idx)
#         train_writer.add_scalar('Loss/Epoch/cd_coarse', avg_cd_coarse, epoch_idx)
#         train_writer.add_scalar('Loss/Epoch/consistency', avg_consistency, epoch_idx)

#         print(
#             '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
#             # (epoch_idx, cfg.train.epochs, epoch_end_time - epoch_start_time,['%.4f' % l for l in [avg_ucd, avg_consistency]]))
#             (epoch_idx, cfg.train.epochs, epoch_end_time - epoch_start_time,
#              ['%.4f' % l for l in [avg_ucd, avg_ucd_coarse, avg_cd_coarse, avg_consistency]]))

#         with open(log_path, "a") as file_object:
#             # ✅ Sab kuch ek hi line mein bracket ke andar
#             file_object.write(f"##########EPOCH {epoch_idx:04d}##########\\n")
            
#             # ✅ Is line ko bhi check karle, ek hi line mein honi chahiye
#             log_msg = f"Training UCD UCD_Coarse CD_Coarse Consistency: {avg_ucd:.4f} {avg_ucd_coarse:.4f} {avg_cd_coarse:.4f} {avg_consistency:.4f}\\n"
#             file_object.write(log_msg)
            
#         # Validate the current model
#         loss_eval = validate(cfg, epoch_idx, val_dataset, val_data_loader, val_writer, model, source_model, test=False)

#         # Save checkpoints
#         if epoch_idx % cfg.train.save_freq == 0 or loss_eval < best_metrics:
#             file_name = 'ckpt-best.pth' if loss_eval < best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
#             output_path = os.path.join(cfg.train.checkpoints, file_name)
#             torch.save({
#                 'epoch_index': epoch_idx,
#                 'best_metrics': best_metrics,
#                 'model': model.state_dict()
#             }, output_path)
#             file_name = 'ckpt-best-source.pth' if loss_eval < best_metrics else 'ckpt-epoch-%03d-source.pth' % epoch_idx
#             output_path = os.path.join(cfg.train.checkpoints, file_name)
#             torch.save({
#                 'epoch_index': epoch_idx,
#                 'best_metrics': best_metrics,
#                 'model': source_model.state_dict()
#             }, output_path)

#             print('Saved checkpoint to %s ...' % output_path)
#             if loss_eval < best_metrics:
#                 best_metrics = loss_eval
#                 with open(log_path, "a") as file_object:
#                     file_object.write('Save ckpt-best.pth...................\\n')

#     train_writer.close()
#     val_writer.close()


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

# === 🚀 NEW IMPORT FOR ADVERSARIAL ALIGNMENT ===
from adversarial_alignment.discriminator import CoarsePointDiscriminator
import torch.nn as nn

# 🔥 NAN SHIELD: Safe Native Distance Functions
def get_safe_squared_dist(p1, p2):
    p1_sq = torch.sum(p1**2, dim=-1, keepdim=True)
    p2_sq = torch.sum(p2**2, dim=-1).unsqueeze(1)
    dist = p1_sq + p2_sq - 2 * torch.bmm(p1, p2.transpose(1, 2))
    return torch.clamp(dist, min=1e-7)

def chamfer_distance_native(p1, p2, sqrt=False):
    dist = get_safe_squared_dist(p1, p2)
    if sqrt:
        dist = torch.sqrt(dist)
    min_dist_1 = torch.min(dist, dim=2)[0]
    min_dist_2 = torch.min(dist, dim=1)[0]
    return torch.mean(min_dist_1) + torch.mean(min_dist_2)

def unidirectional_cd_native(p1, p2, sqrt=False):
    dist = get_safe_squared_dist(p1, p2)
    if sqrt:
        dist = torch.sqrt(dist)
    min_dist = torch.min(dist, dim=2)[0]
    return torch.mean(min_dist)
# ---------------------------------------------
# ===============================================

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
        pin_memory=True) # 👈 Ye False hona chahiye

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
        pin_memory=True) # 🔥 FIX: Changed to False

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
        target.load_state_dict(new_sd, strict=False)
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
    scaler = torch.amp.GradScaler('cuda') # H100 ke liye updated syntax
    
    logging.info('✅ Weights loaded and models moved to GPU!')

    optimizer, scheduler = builder.build_opti_sche(model, cfg)
    
    # === ⚖️ Coarse GAN Setup ===
    discriminator = None # Default value taaki aage crash na ho
    alpha_gan = 0.0      # Default weight 0
    
    if getattr(cfg, 'use_gan', False):
        logging.info("⚖️ Initializing Coarse Adversarial Discriminator...")
        discriminator = CoarsePointDiscriminator().to(device)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        criterion_gan = nn.MSELoss().to(device)
        alpha_gan = 0.05
    # ==========================
    
    # 👇 YE BLOCK ADD KAR (PointMAC Lambda Setup):
    if getattr(cfg, 'use_pointmac', False):
        logging.info("🧠 Initializing PointMAC Meta-Learning Lambdas...")
        param_device = next(model.parameters()).device
        lambda_smr = torch.tensor(0.0, device=param_device, requires_grad=True)
        lambda_ad = torch.tensor(0.0, device=param_device, requires_grad=True)
        lambda_optimizer = torch.optim.Adam([lambda_smr, lambda_ad], lr=1e-3)
        meta_weight = 1.0
    # 👆 ==========================================
    
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
        if discriminator is not None:
            discriminator.train()

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

                optimizer.zero_grad()
                if getattr(cfg, 'use_pointmac', False):
                    lambda_optimizer.zero_grad()

                # 🔥 H100 Power Mode Start
                with torch.amp.autocast("cuda", enabled=False):
                    mask_partial = mask_partial.float()
                    
                    # =========================================================
                    # 🚀 TRUE MAML: BI-LEVEL OPTIMIZATION LOGIC
                    # =========================================================
                    if getattr(cfg, 'use_pointmac', False):
                        # --- INNER LOOP ---
                        _, aux_outputs_inner = model(mask_partial.contiguous(), return_aux=True)
                        target_for_aux = mask_partial 
                        
                        mae_rec_inner = aux_outputs_inner['mae_rec']
                        denoise_pred = aux_outputs_inner['denoise_pred']
                        denoise_target = aux_outputs_inner['denoise_target']
                        
                        loss_aux1 = chamfer_distance_native(mae_rec_inner, target_for_aux, sqrt=False) * 1e2
                        loss_aux2 = torch.mean((denoise_pred - denoise_target) ** 2) * 1e2
                        
                        alpha_tilde = torch.log(1 + lambda_smr ** 2)
                        beta_tilde = torch.log(1 + lambda_ad ** 2)
                        w_smr = torch.exp(alpha_tilde) / (torch.exp(alpha_tilde) + torch.exp(beta_tilde))
                        w_ad = 1.0 - w_smr
                        loss_inner = w_smr * loss_aux1 + w_ad * loss_aux2

                        # --- VIRTUAL TTA STEP ---
                        encoder_params = dict(model.feat_extractor.named_parameters())
                        grads = torch.autograd.grad(loss_inner, encoder_params.values(), create_graph=True, allow_unused=True)
                        
                        fast_weights = OrderedDict()
                        inner_lr = 1e-4 
                        for (name, param), grad in zip(encoder_params.items(), grads):
                            fast_weights[name] = (param - inner_lr * grad) if grad is not None else param

                        # --- OUTER LOOP ---
                        student_out, aux_outputs = model(mask_partial.contiguous(), fast_weights=fast_weights, return_aux=True)
                    else:
                        student_out = model(mask_partial.contiguous())
                        loss_inner = torch.tensor(0.0).to(device)
                    # =========================================================
                    
                    pcd_pred = student_out[-1]
                    coarse_pcd = student_out[0]
                    
                    pcd_pred1, pcd_pred2 = pcd_pred[0:bs], pcd_pred[bs:]
                    coarse_pcd1, coarse_pcd2 = coarse_pcd[0:bs], coarse_pcd[bs:]

                    with torch.no_grad():
                        source_out = source_model(partial.contiguous())
                        coarse_source, source_pred = source_out[0], source_out[-1]

                    sample_source = fps_subsample(source_pred, 256)
                    
                    # 🔥 USING NAN SHIELD NATIVE MATH 🔥
                    loss_ucd_coarse = unidirectional_cd_native(sample_source, pcd_pred1, sqrt=False) + unidirectional_cd_native(sample_source, pcd_pred2, sqrt=False)
                    loss_cd_coarse = chamfer_distance_native(coarse_pcd1, coarse_source, sqrt=False) + chamfer_distance_native(coarse_pcd2, coarse_source, sqrt=False)
                    loss_complete = unidirectional_cd_native(partial, pcd_pred1, sqrt=False) + unidirectional_cd_native(partial, pcd_pred2, sqrt=False)
                    loss_consistency = chamfer_distance_native(pcd_pred1, pcd_pred2, sqrt=False)

                    # --- STATS EXTRACTION ---
                    ucd = loss_complete.item() * 1e4
                    ucd_coarse = loss_ucd_coarse.item() * 1e4
                    cd_coarse = loss_cd_coarse.item() * 1e4
                    consistency = loss_consistency.item() * 1e4

                    total_ucd += ucd
                    total_ucd_coarse += ucd_coarse
                    total_cd_coarse += cd_coarse
                    total_consistency += consistency
                    
                    loss_main = loss_cd_coarse + loss_complete * 1e2 + loss_ucd_coarse + loss_consistency * 1e2
                    
                    if getattr(cfg, 'use_pointmac', False):
                        loss_total = loss_main + meta_weight * loss_inner
                    else:
                        loss_total = loss_main

                    # =========================================================
                    # 🥊 GAN CODE (Optional)
                    # =========================================================
                    loss_G_adv = torch.tensor(0.0) 
                    if getattr(cfg, 'use_gan', False) and discriminator is not None:
                        coarse_real = coarse_source.detach()
                        coarse_fake = coarse_pcd1
                        
                        valid_labels = torch.ones((bs, 1), dtype=torch.float32, device=device)
                        fake_labels = torch.zeros((bs, 1), dtype=torch.float32, device=device)

                        optimizer_D.zero_grad()
                        loss_D_real = criterion_gan(discriminator(coarse_real), valid_labels)
                        loss_D_fake = criterion_gan(discriminator(coarse_fake.detach()), fake_labels)

                        loss_D = 0.5 * (loss_D_real + loss_D_fake)
                        scaler.scale(loss_D).backward()
                        scaler.step(optimizer_D)
                        
                        loss_G_adv = criterion_gan(discriminator(coarse_fake), valid_labels)
                        loss_total = loss_total + (alpha_gan * loss_G_adv)

                # 🔥 Optimized Backward
                scaler.scale(loss_total).backward() 
                # 🔥 NAN PREVENTION: Clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
                scaler.step(optimizer)
                if getattr(cfg, 'use_pointmac', False):
                    scaler.step(lambda_optimizer) 
                
                scaler.update()
                
                n_itr = (epoch_idx - 1) * n_batches + batch_idx

                train_writer.add_scalar('Loss/Batch/ucd', ucd, n_itr)
                train_writer.add_scalar('Loss/Batch/ucd_coarse', ucd_coarse, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_coarse', cd_coarse, n_itr)
                train_writer.add_scalar('Loss/Batch/consistency', consistency, n_itr)

                batch_time.update(time() - batch_end_time)
                batch_end_time = time()
                
                t.set_description(
                    '[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.train.epochs, batch_idx + 1, n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [ucd, ucd_coarse, cd_coarse, consistency]])
                
                if batch_idx % 20 == 0:
                    logging.info(f"👉 [Epoch {epoch_idx}] Batch {batch_idx}/{n_batches} processed! | UCD: {ucd:.4f} | CD Coarse: {cd_coarse:.4f} | GAN Loss: {loss_G_adv.item():.4f}")
                
                if cfg.scheduler.type == 'GradualWarmup':
                    if n_itr < cfg.scheduler.kwargs_2.total_epoch:
                        scheduler.step()
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