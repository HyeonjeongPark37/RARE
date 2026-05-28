import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.helpers.decode_helper import _transpose_and_gather_feat
from lib.losses.focal_loss import focal_loss_cornernet as focal_loss
from lib.losses.uncertainty_loss import laplacian_aleatoric_uncertainty_loss
from tools.eval import box3d_overlap
import numpy as np
import torch.distributed as dist

#GUPNet
class Hierarchical_Task_Learning:
    def __init__(self,epoch0_loss,stat_epoch_nums=5):
        self.index2term = [*epoch0_loss.keys()]
        self.term2index = {term:self.index2term.index(term) for term in self.index2term}  #term2index
        self.stat_epoch_nums = stat_epoch_nums
        self.past_losses=[]
        self.loss_graph = {'seg_loss':[],
                           'size2d_loss':[], 
                           'offset2d_loss':[],
                           'offset3d_loss':['size2d_loss','offset2d_loss'], 
                           'size3d_loss':['size2d_loss','offset2d_loss'], 
                           'heading_loss':['size2d_loss','offset2d_loss'], 
                           'depth_loss':['size2d_loss','size3d_loss','offset2d_loss'],
                           'point_loss': ['offset3d_loss', 'size3d_loss', 'depth_loss'],
                           'pair_loss': ['offset3d_loss', 'size3d_loss', 'depth_loss'],
                           'balance_loss': ['offset3d_loss', 'size3d_loss', 'depth_loss'],

                        }                                 

    def compute_weight(self, current_loss, epoch):
        T = 140
        epsilon = 1e-6  # to avoid division by 0

        loss_weights = {}
        for term in self.loss_graph:
            loss_weights[term] = torch.tensor(1.0).to(current_loss[term].device) \
                if len(self.loss_graph[term]) == 0 else torch.tensor(0.0).to(current_loss[term].device)

        eval_loss_input = torch.cat([_.unsqueeze(0) for _ in current_loss.values()]).unsqueeze(0)

        if len(self.past_losses) == self.stat_epoch_nums:
            past_loss = torch.cat(self.past_losses)  # [N, #terms]
            mean_diff = (past_loss[:-2] - past_loss[2:]).mean(0)  # [#terms]

            if not hasattr(self, 'init_diff'):
                self.init_diff = mean_diff.clone()
                self.init_diff[self.init_diff.abs() < epsilon] = epsilon

            safe_ratio = mean_diff / self.init_diff
            safe_ratio = torch.nan_to_num(safe_ratio, nan=0.0, posinf=1.0, neginf=-1.0)
            c_weights = 1 - safe_ratio.relu().unsqueeze(0)  # [1, #terms]

            time_value = min(((epoch - self.stat_epoch_nums) / (T - self.stat_epoch_nums)), 1.0)

            for current_topic in self.loss_graph:
                if len(self.loss_graph[current_topic]) > 0:
                    control_weight = 1.0
                    for pre_topic in self.loss_graph[current_topic]:
                        w = c_weights[0][self.term2index[pre_topic]]
                        if not torch.isfinite(w):
                            print(f"[Warning] NaN/Inf in c_weight: {pre_topic} = {w}")
                            w = torch.tensor(0.0, device=w.device)
                        control_weight *= w
                    control_weight = torch.clamp(control_weight, 0.0, 1.0)
                    loss_weights[current_topic] = time_value ** (1.0 - control_weight)
                    if not torch.isfinite(loss_weights[current_topic]):
                        print(f"[Fix] NaN loss weight: {current_topic}")
                        loss_weights[current_topic] = torch.tensor(1.0, device=loss_weights[current_topic].device)

            self.past_losses.pop(0)

        self.past_losses.append(eval_loss_input)
        return loss_weights

    def update_e0(self, eval_loss):
        self.epoch0_loss = torch.cat([_.unsqueeze(0) for _ in eval_loss.values()]).unsqueeze(0)


class RARELoss(nn.Module):
    def __init__(self,epoch, device):
        super().__init__()
        self.stat = {}
        self.epoch = epoch
        self.cls_mean_size = np.array([
            [1.76255119, 0.66068622, 0.84422524],
            [1.52563191462, 1.62856739989, 3.88311640418],
            [1.73698127, 0.59706367, 1.76282397]
        ]) 
                                       
        self.cls_mean_size = torch.tensor(self.cls_mean_size, dtype=torch.float32).to(device)
        self.device = device 


    def forward(self, preds, targets, calib, coord_ranges):
        seg_loss = self.compute_segmentation_loss(preds, targets)
        bbox2d_loss = self.compute_bbox2d_loss(preds, targets)
        bbox3d_loss  = self.compute_bbox3d_loss(preds, targets, calib, coord_ranges)
        
        loss = seg_loss + bbox2d_loss + bbox3d_loss
        
        return loss, self.stat


    def compute_segmentation_loss(self, input, target):
        input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
        loss = focal_loss(input['heatmap'], target['heatmap'])
        self.stat['seg_loss'] = loss
        return loss


    def compute_bbox2d_loss(self, input, target):

        if target['mask_2d'].sum() <= 0:
            size2d_loss = torch.tensor(0.0).to(input['size_2d'].device)
            offset2d_loss = torch.tensor(0.0).to(input['size_2d'].device)
        else:
            # compute size2d loss
            size2d_input = extract_input_from_tensor(input['size_2d'], target['indices'], target['mask_2d'])
            size2d_target = extract_target_from_tensor(target['size_2d'], target['mask_2d'])
            size2d_loss = F.l1_loss(size2d_input, size2d_target, reduction='mean')
            # compute offset2d loss
            offset2d_input = extract_input_from_tensor(input['offset_2d'], target['indices'], target['mask_2d'])
            offset2d_target = extract_target_from_tensor(target['offset_2d'], target['mask_2d'])
            offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='mean')

        loss = offset2d_loss + size2d_loss   
        self.stat['offset2d_loss'] = offset2d_loss
        self.stat['size2d_loss'] = size2d_loss
        return loss


    def compute_bbox3d_loss(self, input, target, calib, coord_ranges, mask_type = 'mask_2d'):        
        if target[mask_type].sum() <= 0:
            depth_loss    = torch.tensor(0.0).to(input['size_3d'].device)
            offset3d_loss = torch.tensor(0.0).to(input['size_3d'].device)
            size3d_loss   = torch.tensor(0.0).to(input['size_3d'].device)
            heading_loss  = torch.tensor(0.0).to(input['size_3d'].device)
            point_loss = torch.tensor(0.0).to(input['size_3d'].device)
            pair_loss = torch.tensor(0.0).to(input['size_3d'].device)
            balance_loss = torch.tensor(0.0).to(input['size_3d'].device)
        else:
            sel_mask = input['train_tag']                          # [M·H] boolean
            num_sel = int(sel_mask.sum().item())                   # M·H
            M = int(target[mask_type].sum().item())                # M
            H = max(1, num_sel // max(M, 1))                       # H (assume M>0)

            depth_all      = input['depth'][sel_mask].view(M, H, -1)             # [M,H,2] (mu, logb)
            off3d_all      = input['offset_3d'][sel_mask].view(M, H, -1)         # [M,H,2]
            size3d_all     = input['size_3d'][sel_mask].view(M, H, -1)           # [M,H,3]
            head_all       = input['heading'][sel_mask].view(M, H, -1)           # [M,H,24] (0..11 cls, 12..23 res)
            h3d_logvar_all = input['h3d_log_variance'][sel_mask].view(M, H, -1)  # [M,H,1]
            iou_conf      = input['iou_conf'][sel_mask].view(M, H, -1)                # [B, Kpred] or [B, Kpred,1]

            # GT
            depth_t  = extract_target_from_tensor(target['depth'],       target[mask_type]).view(M, 1)  # [M,1]
            off3d_t  = extract_target_from_tensor(target['offset_3d'],   target[mask_type]).view(M, 2)  # [M,2]
            size3d_t = extract_target_from_tensor(target['size_3d'],     target[mask_type]).view(M, 3)  # [M,3]
            hb       = extract_target_from_tensor(target['heading_bin'], target[mask_type]).view(M)     # [M]
            hr       = extract_target_from_tensor(target['heading_res'], target[mask_type]).view(M)     # [M]

            mu   = depth_all[..., 0]    # [M,H]
            lvar = depth_all[..., 1]    # [M,H]
            depth_t_b = depth_t.expand(M, H)                                    # [M,H]

            # (1) Depth Laplace NLL per hyp
            depth_nll = 1.4142 * torch.exp(-0.5 * lvar) * (mu - depth_t_b).abs() + 0.5 * lvar  # [M,H]

            # (2) 3D offset L1
            off_l1 = (off3d_all - off3d_t.unsqueeze(1)).abs().mean(-1)          # [M,H]

            # (3) 3D size proxy
            size_l1  = (size3d_all[..., 1:] - size3d_t[..., 1:].unsqueeze(1)).abs().mean(-1)  # [M,H]
            h_pred   = size3d_all[..., 0]                                       # [M,H]
            h_t_b    = size3d_t[..., 0].unsqueeze(1)                            # [M,1]→[M,H]
            h_logvar = h3d_logvar_all[..., 0]                                   # [M,H]
            h_nll    = 1.4142 * torch.exp(-0.5 * h_logvar) * (h_pred - h_t_b).abs() + 0.5 * h_logvar
            size_proxy = (2.0/3.0) * size_l1 + (1.0/3.0) * h_nll                # [M,H]

            # (4) Heading proxy = CE + L1(residual)
            cls_logits = head_all[..., :12].reshape(-1, 12)                      # [M*H,12]
            hb_rep     = hb.view(M, 1).expand(M, H).reshape(-1)                  # [M*H]
            ce = F.cross_entropy(cls_logits, hb_rep, reduction='none').view(M, H)

            residuals = head_all[..., 12:24]                                     # [M,H,12]
            hb_idx    = hb.view(M, 1, 1).expand(M, H, 1)                         # [M,H,1]
            pred_res  = residuals.gather(2, hb_idx).squeeze(2)                   # [M,H]
            reg = (pred_res - hr.view(M, 1)).abs()                               # [M,H]
            
            heading_proxy = ce + reg                                             # [M,H]

            eps = 1e-6
            
            s = iou_conf.squeeze(-1) if iou_conf.dim()==3 else iou_conf     # [M,H] in (0,1)
            z = safe_logit(s.clamp(1e-6, 1-1e-6))                        # logits
            
            p = F.softmax(z, dim=1) 

            p_for_sbom = p.detach()              
            
            
            depth_loss    = (p_for_sbom * depth_nll).sum(1).mean()
            offset3d_loss = (p_for_sbom * off_l1).sum(1).mean()
            size3d_loss   = (p_for_sbom * size_proxy).sum(1).mean()
            heading_loss  = (p_for_sbom * heading_proxy).sum(1).mean()

            
            point_loss, pair_loss = self.confidence_loss(input, target, calib, coord_ranges, score_thresh=0.1, min_height_px=10)
            
            p_for_bal  = p

            p_bar = global_mean_over_samples(p_for_bal, dim=0)   # [H]
            u = torch.full_like(p_bar, 1.0 / p_bar.numel())
            balance_loss = F.l1_loss(p_bar, u, reduction='sum')

        loss = depth_loss + offset3d_loss + size3d_loss + heading_loss + (point_loss * 10) + (pair_loss * 0.5) + balance_loss
            

        if depth_loss != depth_loss:
            print('badNAN----------------depth_loss', depth_loss)
        if offset3d_loss != offset3d_loss:
            print('badNAN----------------offset3d_loss', offset3d_loss)
        if size3d_loss != size3d_loss:
            print('badNAN----------------size3d_loss', size3d_loss)
        if heading_loss != heading_loss:
            print('badNAN----------------heading_loss', heading_loss)
        
        self.stat['depth_loss'] = depth_loss
        self.stat['offset3d_loss'] = offset3d_loss
        self.stat['size3d_loss'] = size3d_loss
        self.stat['heading_loss'] = heading_loss
        self.stat['point_loss'] = point_loss
        self.stat['pair_loss'] = pair_loss
        self.stat['balance_loss'] = balance_loss

        
        return loss


    # ---------- heading helpers ----------
    def get_heading_angle_torch(self, heading_pred: torch.Tensor):
        """
        heading_pred: [N, 24] (first 12: logits, next 12: residuals)
        returns angle [N] in [-pi, pi]
        """
        heading_bin_logits = heading_pred[:, :12]
        heading_residuals  = heading_pred[:, 12:]
        cls = torch.argmax(heading_bin_logits, dim=1)              # [N]
        res = heading_residuals.gather(1, cls.unsqueeze(1)).squeeze(1)
        angle_per_class = 2 * np.pi / 12.0
        angle_center = cls.float() * angle_per_class
        angle = angle_center + res
        angle = torch.where(angle > np.pi, angle - 2 * np.pi, angle)
        angle = torch.where(angle < -np.pi, angle + 2 * np.pi, angle)
        return angle

    def class2angle_tensor(self, cls, residual, to_label_format=False):
        angle_per_class = 2 * np.pi / float(12)
        angle_center = cls * angle_per_class
        angle = angle_center + residual
        if to_label_format:
            if isinstance(angle, torch.Tensor):
                angle = torch.where(angle > np.pi, angle - 2 * np.pi, angle)
                angle = torch.where(angle < -np.pi, angle + 2 * np.pi, angle)
            else:
                if angle > np.pi: angle -= 2 * np.pi
                if angle < -np.pi: angle += 2 * np.pi
        return angle

    def project2rect(self, calib, point_img):
        c_u = calib[:, 0, 2]
        c_v = calib[:, 1, 2]
        f_u = calib[:, 0, 0]
        f_v = calib[:, 1, 1]
        b_x = calib[:, 0, 3] / (-f_u)
        b_y = calib[:, 1, 3] / (-f_v)
        x = (point_img[:, 0] - c_u) * point_img[:, 2] / f_u + b_x
        y = (point_img[:, 1] - c_v) * point_img[:, 2] / f_v + b_y
        z = point_img[:, 2]
        return torch.stack([x,y,z], dim=-1)


    def confidence_loss(self, input, target, calib, coord_ranges, mask_type='mask_2d', score_thresh=0.2, min_height_px=25):
        # ===== basics =====
        heatmap = input['heatmap']                       # [B,C,Hf,Wf]
        B, C, Hf, Wf = heatmap.shape
        device = heatmap.device

        indices    = input['indices_conf']               # [B, Kpred]
        cls_ids_dt = input['cls_ids_conf']               # [B, Kpred]
        assert indices.dim() == 2, "indices_conf must be [B, Kpred]"
        Kpred = indices.shape[1]

        # K/H 
        Kgt = target['indices'].shape[1]
        assert Kpred % Kgt == 0, f"Kpred({Kpred}) must be multiple of Kgt({Kgt})"
        Hhyp = Kpred // Kgt

        calib = calib.to(device)

        # ---- helpers ----
        def BK(x):
            x = x.to(device)
            if x.dim() >= 2 and x.shape[0] == B and x.shape[1] == Kpred:
                return x
            if x.dim() >= 2 and x.shape[0] == B * Kpred:
                return x.view(B, Kpred, *x.shape[1:])
            if x.dim() == 1 and x.shape[0] == B * Kpred:
                return x.view(B, Kpred)
            raise RuntimeError(f"Unexpected BK shape {tuple(x.shape)}")

        def expand_to_kpred(x, name="gt"):
            x = x.to(device)
            if x.dim() == 2:
                assert x.shape[0] == B and x.shape[1] == Kgt, f"{name} must be [B,{Kgt}]"
                return x.unsqueeze(2).expand(B, Kgt, Hhyp).reshape(B, Kpred)
            elif x.dim() == 3:
                assert x.shape[0] == B and x.shape[1] == Kgt, f"{name} must be [B,{Kgt},D]"
                D = x.shape[2]
                return x.unsqueeze(2).expand(B, Kgt, Hhyp, D).reshape(B, Kpred, D)
            else:
                raise RuntimeError(f"{name} must be [B,K] or [B,K,D], got {tuple(x.shape)}")

        # ---- predictions ----
        depth            = BK(input['depth_conf'])           # [B,Kpred,1 or 2]
        offset3d_input   = BK(input['offset_3d_conf'])       # [B,Kpred,2]
        size3d_input     = BK(input['size_3d_conf'])         # [B,Kpred,3] (delta_hwl)
        heading_raw      = BK(input['heading_conf'])         # [B,Kpred,24]
        bb_3d_logit      = BK(input['bb_3d_logit'])          # [B,Kpred] or [B,Kpred,1]
        if bb_3d_logit.dim() == 3:
            bb_3d_logit = bb_3d_logit.squeeze(-1)
        cls_ids_dt = BK(cls_ids_dt).long()
        if cls_ids_dt.dim() == 3:
            cls_ids_dt = cls_ids_dt.squeeze(-1)

        # alpha
        alpha_pred = self.get_heading_angle_torch(heading_raw.reshape(B * Kpred, -1)).view(B, Kpred, 1)  # [-pi,pi]

        # mean size(h,w,l) → decode to (l,h,w)
        mean_sizes_dt = self.cls_mean_size[cls_ids_dt]                 # [B,Kpred,3]

        # grid → ROI coords
        xs = (indices % Wf).float() / Wf
        ys = (indices // Wf).float() / Hf
        cr = coord_ranges.to(device).unsqueeze(1).expand(B, Kpred, 2, 2)
        x_min, y_min = cr[:, :, 0, 0], cr[:, :, 0, 1]
        x_max, y_max = cr[:, :, 1, 0], cr[:, :, 1, 1]
        xs = xs * (x_max - x_min) + x_min
        ys = ys * (y_max - y_min) + y_min
        scale_x = (x_max - x_min) / Wf
        scale_y = (y_max - y_min) / Hf

        
        xs3d = xs + offset3d_input[..., 0] * scale_x
        ys3d = ys + offset3d_input[..., 1] * scale_y
        depth_scalar = depth[..., 0] if depth.size(-1) >= 1 else depth.squeeze(-1)
        uvd_flat = torch.stack([xs3d, ys3d, depth_scalar], dim=-1).view(B * Kpred, 3)
        calib_rep = calib.repeat_interleave(Kpred, dim=0)
        loc_flat = self.project2rect(calib_rep, uvd_flat)             # [B*Kpred,3]
        locations_dt = loc_flat.view(B, Kpred, 3)

        # size decode & y(bottom)→y(center)
        dimensions_dt = (size3d_input + mean_sizes_dt)[:, :, [2, 1, 0]]  # (l,h,w)
        locations_dt[:, :, 1] += dimensions_dt[:, :, 1] / 2.0            # h index=1 after reorder

        # ===== alpha → ry (예측) =====
        pi = np.pi
        ry_dt = alpha_pred.squeeze(-1) + torch.atan2(locations_dt[..., 0], locations_dt[..., 2])
        ry_dt = (ry_dt + pi) % (2 * pi) - pi
        ry_dt = ry_dt.unsqueeze(-1)

        pred_box3d = torch.cat([locations_dt, dimensions_dt, ry_dt], dim=-1)  # [B,Kpred,7]

        # GT
        mask2d          = expand_to_kpred(target[mask_type],     "mask_2d").bool()
        size3d_t_full   = expand_to_kpred(target['size_3d'],     "size_3d")
        offset3d_t_full = expand_to_kpred(target['offset_3d'],   "offset_3d")
        depth_t_full    = expand_to_kpred(target['depth'],       "depth").squeeze(-1)
        cls_ids_gt_full = expand_to_kpred(target['cls_ids'],     "cls_ids").long()
        heading_bin_t   = expand_to_kpred(target['heading_bin'], "heading_bin").long()
        heading_res_t   = expand_to_kpred(target['heading_res'], "heading_res")
        indices_gt      = expand_to_kpred(target['indices'],     "indices")
        if heading_bin_t.dim() == 3: heading_bin_t = heading_bin_t.squeeze(-1)
        if heading_res_t.dim() == 3: heading_res_t = heading_res_t.squeeze(-1)

        
        heading_target_full = torch.zeros(B, Kpred, 1, device=device, dtype=pred_box3d.dtype)
        hb = heading_bin_t[mask2d]
        hr = heading_res_t[mask2d]
        if hb.numel() > 0:
            ht = self.class2angle_tensor(hb, hr, True).to(device)  # alpha
            heading_target_full[mask2d] = ht.unsqueeze(-1)

        
        xs_gt = ((indices_gt % Wf).float() / Wf).to(device)
        ys_gt = ((indices_gt // Wf).float() / Hf).to(device)
        xs_gt = xs_gt * (x_max - x_min) + x_min
        ys_gt = ys_gt * (y_max - y_min) + y_min
        scale_x_gt = (x_max - x_min) / Wf
        scale_y_gt = (y_max - y_min) / Hf

        
        # DT(height) from prediction @ Kpred
        size2d_pred   = _transpose_and_gather_feat(input['size_2d'], indices)   # [B,Kpred,2]
        height_dtpx   = size2d_pred[..., 1].clamp(min=0) * scale_y              # [B,Kpred]
        dt_h_mask     = height_dtpx > float(min_height_px)                      # [B,Kpred]

        # GT(height) from target['size_2d'] @ K → expand to Kpred
        K_tmp = Kgt
        cr_gt_tmp = coord_ranges.to(device).unsqueeze(1).expand(B, K_tmp, 2, 2)     # [B,K,2,2]
        y_min_gt_tmp, y_max_gt_tmp = cr_gt_tmp[:, :, 0, 1], cr_gt_tmp[:, :, 1, 1]
        scale_y_gt_tmp = (y_max_gt_tmp - y_min_gt_tmp) / Hf                         # [B,K]
        size2d_gt_tmp   = target['size_2d'].to(device)                               # [B,K,2]
        height_gtpx_tmp = size2d_gt_tmp[..., 1].clamp(min=0) * scale_y_gt_tmp        # [B,K]
        gt_h_mask_K     = height_gtpx_tmp > float(min_height_px)                     # [B,K]
        gt_h_mask       = gt_h_mask_K.unsqueeze(2).expand(B, K_tmp, Hhyp).reshape(B, Kpred)  # [B,Kpred]

        # GT 3D box (bottom→center)
        xs3d_gt_full = xs_gt + offset3d_t_full[..., 0] * scale_x_gt
        ys3d_gt_full = ys_gt + offset3d_t_full[..., 1] * scale_y_gt
        uvd_gt_flat  = torch.stack([xs3d_gt_full, ys3d_gt_full, depth_t_full], dim=-1).view(B * Kpred, 3)
        loc_gt_flat  = self.project2rect(calib_rep, uvd_gt_flat)
        locations_gt_full = loc_gt_flat.view(B, Kpred, 3)

        mean_sizes_gt_full = self.cls_mean_size[cls_ids_gt_full].to(device)          # [B,Kpred,3]
        dimensions_gt_full = (size3d_t_full + mean_sizes_gt_full)[:, :, [2, 1, 0]]   # (l,h,w)
        locations_gt_full[:, :, 1] += dimensions_gt_full[:, :, 1] / 2.0              # h index=1

        # ===== alpha → ry (GT) =====
        alpha_gt = heading_target_full.squeeze(-1)  # [B,Kpred]
        ry_gt = alpha_gt + torch.atan2(locations_gt_full[..., 0], locations_gt_full[..., 2])
        ry_gt = (ry_gt + pi) % (2 * pi) - pi
        ry_gt = ry_gt.unsqueeze(-1)

        gt_box3d_full = torch.cat([locations_gt_full, dimensions_gt_full, ry_gt], dim=-1)  # [B,Kpred,7]

        # 2d score filtering
        hm   = torch.clamp(heatmap.sigmoid(), 1e-4, 1 - 1e-4)
        hm_g = _transpose_and_gather_feat(hm, indices)                         # [B,Kpred,C]
        scores_k = hm_g.gather(2, cls_ids_dt.unsqueeze(-1)).squeeze(-1)        # [B,Kpred]
        score_mask = scores_k > float(score_thresh)

        # pseudo IoU (one-to-many, class-wise)
        assigned_mask = torch.zeros(B, Kpred, device=device, dtype=torch.bool)
        pseudo_conf   = torch.zeros(B, Kpred, device=device, dtype=bb_3d_logit.dtype)

        for b in range(B):
            m_gt_b = (mask2d[b] & gt_h_mask[b])          
            if not m_gt_b.any():
                continue
            gt_b     = gt_box3d_full[b][m_gt_b]
            gt_cls_b = cls_ids_gt_full[b][m_gt_b]
            
            for c in gt_cls_b.unique():
                m_dt = (cls_ids_dt[b] == c)
                if not m_dt.any():
                    continue
                pred_b_c = pred_box3d[b][m_dt]
                gt_b_c   = gt_b[gt_cls_b == c]
                if pred_b_c.numel() == 0 or gt_b_c.numel() == 0:
                    continue

                with torch.no_grad():
                    ious = box3d_overlap(
                        pred_b_c.detach().cpu().numpy(),
                        gt_b_c.detach().cpu().numpy()
                    )  # [n_dt, n_gt_c]
                    iou_max = ious.max(axis=1)

                pseudo_conf[b, m_dt]   = torch.as_tensor(iou_max, device=pseudo_conf.device, dtype=pseudo_conf.dtype)
                assigned_mask[b, m_dt] = True

        valid_mask = assigned_mask & score_mask & dt_h_mask  

        # --------- losses ---------
        if valid_mask.any():
            p_hat_list  = bb_3d_logit[valid_mask].reshape(-1)
            p_star_list = pseudo_conf[valid_mask].reshape(-1)

            delta_ps_min      = 0.1
            max_pairs_per_cls = 32

            L_mse_terms, L_rank_terms = [], []

            for b in range(B):
                vb = valid_mask[b]
                if not vb.any():
                    continue

                p_hat_b  = bb_3d_logit[b][vb]
                p_star_b = pseudo_conf[b][vb].detach()
                cls_b    = cls_ids_dt[b][vb]
                if p_hat_b.numel() == 0:
                    continue

                # point-wise absolute ranking loss (MSE)
                mse_elem = F.mse_loss(p_hat_b, p_star_b, reduction='none')
                w = torch.where(p_star_b < 0.1, torch.full_like(p_star_b, 0.1), torch.ones_like(p_star_b))
                L_mse_terms.append((w * mse_elem).mean())

                # class-wise pairwise ranking (logit space)
                for c in cls_b.unique():
                    mc = (cls_b == c)
                    if mc.sum() < 2:
                        continue
                    p_hat_c  = p_hat_b[mc]
                    p_star_c = p_star_b[mc]
                    if (p_star_c.max() == p_star_c.min()):
                        continue

                    N  = p_hat_c.numel()
                    dp = p_star_c.view(N, 1) - p_star_c.view(1, N)
                    mask_valid_pairs = (dp.abs() >= delta_ps_min) & (~torch.eye(N, dtype=bool, device=p_hat_c.device))
                    pairs = mask_valid_pairs.nonzero(as_tuple=False)
                    M = pairs.shape[0]
                    if M == 0:
                        continue

                    take   = min(max_pairs_per_cls, M)
                    perm   = torch.randperm(M, device=p_hat_c.device)[:take]
                    chosen = pairs[perm]
                    i, j   = chosen[:, 0], chosen[:, 1]

                    s_logit    = safe_logit(p_hat_c)                 # p ∈ (0,1) → logits
                    y_ij       = (p_star_c[i] > p_star_c[j]).float()
                    logit_diff = s_logit[i] - s_logit[j]
                    L_rank_terms.append(F.binary_cross_entropy_with_logits(logit_diff, y_ij, reduction='mean'))

            point_loss       = torch.stack(L_mse_terms).mean()  if L_mse_terms  else (bb_3d_logit.mean() * 0)
            pair_loss = torch.stack(L_rank_terms).mean() if L_rank_terms else (bb_3d_logit.mean() * 0)
        else:
            point_loss       = torch.tensor(0.0, device=bb_3d_logit.device)
            pair_loss = torch.tensor(0.0, device=bb_3d_logit.device)
            
        return point_loss, pair_loss



### ======================  auxiliary functions  =======================

def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask.bool()]  # B*K*C --> M * C


def extract_target_from_tensor(target, mask):
    return target[mask.bool()]

def compute_heading_loss(input, mask, target_cls, target_reg):
    mask = mask.view(-1)           # B*K
    target_cls = target_cls.view(-1)
    target_reg = target_reg.view(-1)

    input_cls = input[:, 0:12]
    target_cls = target_cls[mask]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')
    
    input_reg = input[:, 12:24]
    target_reg = target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12, device=input.device).scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')
    return cls_loss + reg_loss

def safe_logit(p, eps=1e-6):
    p = p.clamp(eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)   

def compute_heading_loss_selected(input_logits, target_cls, target_res):
        """
        input_logits: [N, 24]  (0..11: cls logits, 12..23: residuals)
        target_cls  : [N]      (bin idx)
        target_res  : [N]      (residual)
        """
        input_cls = input_logits[:, 0:12]
        cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

        input_reg = input_logits[:, 12:24]
        onehot = torch.zeros(input_logits.size(0), 12, device=input_logits.device)
        onehot.scatter_(1, target_cls.view(-1,1), 1)
        input_reg = (input_reg * onehot).sum(1)
        reg_loss = F.l1_loss(input_reg, target_res, reduction='mean')
        return cls_loss + reg_loss

def global_mean_over_samples(p: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    p: [M, H] 같은 텐서에서 샘플 차원(dim) 기준 전-프로세스(DDP world) 평균을 구함.
    싱글/DP(비-DDP)에서는 로컬 mean과 동일하게 동작.
    """
    local_sum = p.sum(dim=dim)  # [H]
    local_cnt = torch.tensor(p.size(dim), device=p.device, dtype=local_sum.dtype)  # scalar tensor

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_cnt, op=dist.ReduceOp.SUM)

    # 텐서 나눗셈으로 CPU 동기화 회피, 0 분모 방지
    return local_sum / local_cnt.clamp_min(1)