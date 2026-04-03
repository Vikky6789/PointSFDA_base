#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Sanvik

import torch
import torch.nn as nn
import torch.nn.functional as F # Added for PointMAC ITSI
from torch.func import functional_call # 🔥 Required for MAML fast_weights
from SnowflakeNet.SnowflakeNet_utils import PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, Transformer
from SnowflakeNet.skip_transformer import SkipTransformer

# ==========================================================================================
# 🚀 POINTMAC MODULAR COMPONENTS START (Auxiliary Heads for MAML & TTA)
# ==========================================================================================

class DecoderFC(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False):
        super(DecoderFC, self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim

        model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):
            fc_layer = nn.Linear(prev_nf, nf)
            model.append(fc_layer)
            if bn:
                bn_layer = nn.BatchNorm1d(nf)
                model.append(bn_layer)
            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], output_pts*3)
        model.append(fc_layer)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        x = x.view((-1, 3, self.output_pts))
        return x

class ITSI(nn.Module):
    def __init__(self, in_dim=512, latent_dim=128, token_channels=64, num_tokens=1024):
        super(ITSI, self).__init__()
        self.latent_fc = nn.Linear(in_dim, latent_dim)
        self.token_fc = nn.Sequential(
            nn.Linear(in_dim, token_channels * num_tokens),
            nn.BatchNorm1d(token_channels * num_tokens),
        )
        self.token_channels = token_channels
        self.num_tokens = num_tokens

    def forward(self, global_feat):
        latent = self.latent_fc(global_feat)
        tokens = self.token_fc(global_feat)
        tokens = F.relu(tokens).view(-1, self.token_channels, self.num_tokens)
        return latent, tokens

class ExtendedModel(nn.Module):
    """ Auxsmr head: Stochastic Masked Reconstruction branch """
    def __init__(self, original_model, dim_feat=512):
        super(ExtendedModel, self).__init__()
        # ADAPTED: Direct link to SnowflakeNet's FeatureExtractor
        self.feat_extractor_ex = original_model.feat_extractor
        self.mask_ratio = 0.6
        self.number_fine = 8192
        self.itsi = ITSI(in_dim=dim_feat, latent_dim=128, token_channels=64, num_tokens=1024)
        self.decoder = DecoderFC(n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _random_mask_points(self, x):
        B, N, C = x.shape
        if not self.training: return x
        
        # 🛡️ SAFETY CHECK: SnowflakeNet needs at least 512 points. 
        N_keep = max(512, int(N * (1.0 - self.mask_ratio)))
        
        # If input cloud is already tiny, skip masking and pad
        if N < 512:
            idx = torch.cat([torch.arange(N, device=x.device), torch.randint(0, N, (512 - N,), device=x.device)])
            return x[:, idx, :]
        
        noise = torch.rand(B, N, device=x.device)
        ids_keep = torch.topk(noise, k=N_keep, dim=1, largest=False)[1]
        ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, C)
        return torch.gather(x, dim=1, index=ids_keep_expanded)

    def forward(self, x, fast_weights=None):
        x_visible = self._random_mask_points(x)
        pc_x = x_visible.permute(0, 2, 1).contiguous()
        
        # 🔥 Fast weights processing for Inner Loop
        if fast_weights is not None:
            z3 = functional_call(self.feat_extractor_ex, fast_weights, (pc_x,))
        else:
            z3 = self.feat_extractor_ex(pc_x) 
            
        global_feat = torch.max(z3, dim=2)[0]
        z_latent, _ = self.itsi(global_feat)
        return self.decoder(z_latent).transpose(1, 2).contiguous()

class SelfAttentionUnit(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionUnit, self).__init__()
        self.to_q = nn.Sequential(nn.Conv1d(3 + in_channels, 2 * in_channels, 1, bias=False), nn.BatchNorm1d(2 * in_channels), nn.ReLU(inplace=True))
        self.to_k = nn.Sequential(nn.Conv1d(3 + in_channels, 2 * in_channels, 1, bias=False), nn.BatchNorm1d(2 * in_channels), nn.ReLU(inplace=True))
        self.to_v = nn.Sequential(nn.Conv1d(3 + in_channels, 2 * in_channels, 1, bias=False), nn.BatchNorm1d(2 * in_channels), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(nn.Conv1d(2 * in_channels, in_channels, 1, bias=False), nn.BatchNorm1d(in_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        attention_map = torch.matmul(q.permute(0, 2, 1), k)
        value = torch.matmul(attention_map, v.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        return self.fusion(value)

class OffsetRegression(nn.Module):
    def __init__(self, in_channels):
        super(OffsetRegression, self).__init__()
        self.coordinate_regression = nn.Sequential(nn.Conv1d(in_channels, 256, 1), nn.ReLU(inplace=True), nn.Conv1d(256, 64, 1), nn.ReLU(inplace=True), nn.Conv1d(64, 3, 1), nn.Sigmoid())
        self.range_max = 0.5
    def forward(self, x):
        offset = self.coordinate_regression(x)
        offset = offset * self.range_max * 2 - self.range_max
        return offset.permute(0, 2, 1).contiguous()

class ExtendedModel2(nn.Module):
    """ Auxad head: Artifact Denoising branch """
    def __init__(self, original_model, aux1M=None):
        super(ExtendedModel2, self).__init__()
        self.feat_extractor_ex = original_model.feat_extractor
        if aux1M is not None and hasattr(aux1M, "itsi"):
            self.itsi = aux1M.itsi
        else:
            self.itsi = ITSI(in_dim=512, latent_dim=128, token_channels=64, num_tokens=1024)
        self.sa = SelfAttentionUnit(in_channels=61)
        self.offset = OffsetRegression(in_channels=61)

    def forward(self, pts, add_noise=False, fast_weights=None):
        device = pts.device
        
        # 🛡️ SAFETY FIX: Ensure input has enough points for SA layers
        if pts.shape[1] < 512:
            idx_pad = torch.randint(0, pts.shape[1], (512 - pts.shape[1],), device=device)
            pts = torch.cat([pts, pts[:, idx_pad, :]], dim=1)

        process_pts = pts + torch.randn_like(pts) * 0.015 if add_noise else pts
        pc_x = process_pts.permute(0, 2, 1).contiguous()
        
        if fast_weights is not None:
            z3 = functional_call(self.feat_extractor_ex, fast_weights, (pc_x,))
        else:
            z3 = self.feat_extractor_ex(pc_x)
            
        global_feat = torch.max(z3, dim=2)[0]
        _, group_input_tokens = self.itsi(global_feat)
        t = self.sa(group_input_tokens)
        pred_offsets = self.offset(t)
        
        # Sync sampling to match pred_offsets size
        num_avail = process_pts.shape[1]
        num_sample = min(1024, num_avail)
        idx = torch.randperm(num_avail, device=device)[:num_sample]
        base_points = process_pts[:, idx, :]
        
        refine = base_points + pred_offsets[:, :num_sample, :]
        return refine.contiguous(), base_points

# ==========================================================================================
# 🛑 POINTMAC MODULAR COMPONENTS END
# ==========================================================================================

class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=1024):
        """Encoder that encodes information of partial point cloud
        """
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, point_cloud):
        """
        Args:
             point_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l1_points = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l2_points = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)

        return l3_points
        #return [l1_points,l2_points,l3_points]


class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):#(B,512,1)
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))  # (b,128,256)
        x2 = self.mlp_2(x1)#(b,128,256)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        completion = self.mlp_4(x3)  # (b, 3, 256)
        return completion


class SPD(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1):
        """Snowflake Point Deconvolution"""
        super(SPD, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])#Shared MLP
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + dim_feat, layer_dims=[256, 128])

        self.skip_transformer = SkipTransformer(in_channel=128, dim=64)

        self.mlp_ps = MLP_CONV(in_channel=128, layer_dims=[64, 32])
        self.ps = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias=False)   # point-wise splitting

        self.up_sampler = nn.Upsample(scale_factor=up_factor)#采用三线性差值实现的上采样
        self.mlp_delta_feature = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)

        self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[64, 3])

    def forward(self, pcd_prev, feat_global, K_prev=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, dim_feat, 1)
            K_prev: Tensor, (B, 128, N_prev)

        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape#[16, 3, 512]      #[16, 3, 512]       #[16, 3, 2048]
        feat_1 = self.mlp_1(pcd_prev)#[16, 128, 512]    #[16, 128, 512]     #[16, 128, 2048]
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            feat_global.repeat(1, 1, feat_1.size(2))], 1)
        Q = self.mlp_2(feat_1)
        #print(feat_1.shape,Q.shape)
        H = self.skip_transformer(pcd_prev, K_prev if K_prev is not None else Q, Q)
        # [16, 128, 512]     #[16, 128, 512]     #[16, 128, 2048]
        feat_child = self.mlp_ps(H)     #[16, 32, 512]      #[16, 32, 512]      #[16, 32, 2048]
        feat_child = self.ps(feat_child)#[16, 128, 512]     #[16, 128, 2048]    #[16, 128, 16384]
        # (B, 128, N_prev * up_factor)

        H_up = self.up_sampler(H)       #[16, 128, 512]     #[16, 128, 2048]       #[16, 128, 16384]
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))#[16, 128, 512]    #[16, 128, 2048]    #[16, 128, 16384]
        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius**self.i  # (B, 3, N_prev * up_factor)
        #[16, 3, 512]   #[16, 3, 2048]  #[16, 3, 16384]
        #delta为预测的位移
        #这里的radius好像没有用，可能本来是想对位移进行归一化的
        pcd_child = self.up_sampler(pcd_prev) #[16, 3, 512]     #[16, 3, 2048]     #[16, 3, 16384]
        pcd_child = pcd_child + delta

        return pcd_child, K_curr


class Decoder(nn.Module):#(B,512,1)
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512, radius=1, up_factors=None):
        super(Decoder, self).__init__()
        self.num_p0 = num_p0
        self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_pc)
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = [1] + up_factors#上采样率为1 4 8 ，4 8 来自与up_factors

        uppers = []
        for i, factor in enumerate(up_factors):
            uppers.append(SPD(dim_feat=dim_feat, up_factor=factor, i=i, radius=radius))

        self.uppers = nn.ModuleList(uppers)

    def forward(self, feat, partial, return_P0=False):
        """
        Args:
            feat: Tensor, (b, dim_feat, n)
            partial: Tensor, (b, n, 3)
        """
        arr_pcd = []
        pcd = self.decoder_coarse(feat).permute(0, 2, 1).contiguous()  # (B, num_pc, 3) # (b, 256, 3)
        arr_pcd.append(pcd)
        #tmp = torch.cat([pcd, partial],1) #(b,2304,3)
        pcd = fps_subsample(torch.cat([pcd, partial], 1), self.num_p0) # (b,512,3)
        if return_P0:
            arr_pcd.append(pcd)
        K_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous()
        for upper in self.uppers:
            pcd, K_prev = upper(pcd, feat, K_prev)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

        return arr_pcd


class SnowflakeNet(nn.Module):
    def __init__(self, config):
        super(SnowflakeNet, self).__init__()
        self.feat_extractor = FeatureExtractor(out_dim=config.dim_feat)
        self.decoder = Decoder(dim_feat=config.dim_feat, num_pc=config.num_pc, num_p0=config.num_p0, radius=config.radius, up_factors=config.up_factors)

        # ==================================================
        # 🎛️ POINTMAC SWITCH
        # ==================================================
        self.use_pointmac = getattr(config, 'use_pointmac', False)
        
        if self.use_pointmac:
            print("🟢 PointMAC MAML Mode Activated! Loading Bi-Aux Units...")
            self.mae_aux = ExtendedModel(self, dim_feat=config.dim_feat)
            self.denoise_aux = ExtendedModel2(self, aux1M=self.mae_aux)

    def forward(self, point_cloud, return_P0=False, return_aux=False, adapt_mode=False, fast_weights=None):
        pcd_bnc = point_cloud
        pc_permuted = point_cloud.permute(0, 2, 1).contiguous()

        if fast_weights is not None:
            feat = functional_call(self.feat_extractor, fast_weights, (pc_permuted,))
        else:
            feat = self.feat_extractor(pc_permuted)
            
        out = self.decoder(feat, pcd_bnc, return_P0=return_P0)

        if self.use_pointmac and return_aux:
            aux_outputs = {}
            aux_outputs['mae_rec'] = self.mae_aux(pcd_bnc, fast_weights=fast_weights)
            aux_outputs['denoise_pred'], aux_outputs['denoise_target'] = self.denoise_aux(pcd_bnc, add_noise=(self.training or adapt_mode), fast_weights=fast_weights)
            return out, aux_outputs

        return out