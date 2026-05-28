import numpy as np
import torch
import torch.nn as nn
from lib.datasets.utils import class2angle

def decode_detections(dets_pack, info, calibs, cls_mean_size, threshold, use_iou_conf=True):
    B, K, H = dets_pack['depth'].shape[:3]

    results = {}
    
    # pre-fetch
    cls_ids  = dets_pack['cls_ids']     # [B,K]
    scores0  = dets_pack['scores']      # [B,K]
    xs2d_raw = dets_pack['xs2d']        # [B,K]  
    ys2d_raw = dets_pack['ys2d']        # [B,K]
    xs_raw = dets_pack['xs_raw']
    ys_raw = dets_pack['ys_raw']

    size_2d  = dets_pack['size_2d']     # [B,K,2]

    heading  = dets_pack['heading']     # [B,K,H,24]
    size3d   = dets_pack['size3d']      # [B,K,H,3]
    offset3d = dets_pack['offset3d']    # [B,K,H,2]
    depth    = dets_pack['depth']       # [B,K,H,2] (mu, logvar)

    # confidence per hyp
    if use_iou_conf and dets_pack['iou_conf'] is not None:
        conf_BKH = dets_pack['iou_conf']         # [B,K,H], already in [0,1] if trained so
    else:
        logvar = depth[..., 1]                    # [B,K,H]
        conf_BKH = np.exp(-np.exp(0.5 * logvar))

    for i in range(B):
        preds = []
        
        ratio_x = info['bbox_downsample_ratio'][i][0]
        ratio_y = info['bbox_downsample_ratio'][i][1]

        for j in range(K):
            # ---- (1) best h by confidence ----
            conf_h = conf_BKH[i, j, :]                  # [H]
            best_h = int(np.argmax(conf_h))

            score = float(scores0[i, j])
            if score < threshold:
                continue
            
            score *= conf_h[best_h]
            
            x = float(xs2d_raw[i, j] * ratio_x)
            y = float(ys2d_raw[i, j] * ratio_y)
            w = float(size_2d[i, j, 0] * ratio_x)
            h = float(size_2d[i, j, 1] * ratio_y)
            bbox = [x - w/2, y - h/2, x + w/2, y + h/2]

            # heading -> alpha, ry
            heading_h = heading[i, j, best_h, :]        # [24]
            alpha = get_heading_angle(heading_h)
            ry = calibs[i].alpha2ry(alpha, x)

            # ---- (4) pick hyp-specific tensors ----
            dims  = size3d[i, j, best_h, :]         # [3]
            cls_id = int(cls_ids[i, j])
            dimensions = dims + cls_mean_size[cls_id]
            if True in (dimensions<0.0): continue

            off3d_h   = offset3d[i, j, best_h, :]       # [2]
            x3d = float((xs_raw[i, j] + off3d_h[0]) * ratio_x)
            y3d = float((ys_raw[i, j] + off3d_h[1]) * ratio_y)

            x3d_arr   = np.asarray([x3d],   dtype=np.float32)
            y3d_arr   = np.asarray([y3d],   dtype=np.float32)
            
            depth_h   = float(depth[i, j, best_h, 0])   # mu
            depth_arr = np.asarray([depth_h], dtype=np.float32)

            locations = calibs[i].img_to_rect(x3d_arr, y3d_arr, depth_arr).reshape(-1)
            locations[1] += dimensions[0] / 2.0
            preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])

        results[info['img_id'][i]] = preds

    return results
    
def extract_dets_from_outputs(outputs, K=50, to_numpy=True):
    # src
    heatmap = outputs['heatmap']
    size_2d = outputs['size_2d']
    offset_2d = outputs['offset_2d']

    B, C, Hf, Wf = heatmap.size()

    # --- heatmap top-K (query) ---
    heatmap = _nms(torch.clamp(heatmap.sigmoid_(), 1e-4, 1 - 1e-4))
    scores, inds, cls_ids, xs, ys = _topk(heatmap, K=K)

    # 2D offsets/sizes
    offset_2d = _transpose_and_gather_feat(offset_2d, inds).view(B, K, 2)
    size_2d   = _transpose_and_gather_feat(size_2d, inds).view(B, K, 2)
    xs2d = xs.view(B, K, 1) + offset_2d[..., 0:1]
    ys2d = ys.view(B, K, 1) + offset_2d[..., 1:2]

    # helper: map [B, K*H, D] or [B*K*H, D] -> [B,K,H,D]
    def to_BKH(x):
        if x.dim() == 4 and x.shape[0] == B:
            return x
        if x.dim() == 3 and x.shape[0] == B:
            KH = x.shape[1]
            assert KH % K == 0, f"Cannot infer H from {tuple(x.shape)} with K={K}"
            Hhyp = KH // K
            return x.view(B, K, Hhyp, x.shape[2])
        if x.dim() == 2:
            if x.shape[0] == B:
                KH = x.shape[1]
                assert KH % K == 0
                Hhyp = KH // K
                return x.view(B, K, Hhyp, 1)
            else:
                D = x.shape[1] if x.shape[1] > 1 else 1
                Hhyp = (x.shape[0] // B) // K
                return x.view(B, K, Hhyp, D)
        raise RuntimeError(f"Unsupported shape for to_BKH: {tuple(x.shape)}")

    heading_BKH = to_BKH(outputs['heading'])        # [B,K,H,24]
    size3d_BKH  = to_BKH(outputs['size_3d'])        # [B,K,H,3 or 4]
    if size3d_BKH.shape[-1] == 4:
        size3d_BKH = size3d_BKH[..., :3]
    offset3d_BKH = to_BKH(outputs['offset_3d'])     # [B,K,H,2]
    depth_BKH    = to_BKH(outputs['depth'])         # [B,K,H,2] (mu, logvar)

    # optional iou conf branch
    iou_conf_BKH = None
    if 'bb_3d_logit' in outputs:
        iou = outputs['bb_3d_logit']
        if iou.dim() == 3 and iou.shape[-1] == 1:
            iou = iou.squeeze(-1)
        iou = to_BKH(iou)                           # [B,K,H,1] or [B,K,H]
        if iou.dim() == 4 and iou.shape[-1] == 1:
            iou = iou.squeeze(-1)                   # [B,K,H]
        iou_conf_BKH = iou

    dets_pack = {
        'cls_ids': cls_ids.view(B, K).float(),
        'scores': scores.view(B, K),
        'xs2d': xs2d.view(B, K),
        'ys2d': ys2d.view(B, K),
        'xs_raw': xs.view(B, K),  # ← 추가: 3D 투영용 원본 xs
        'ys_raw': ys.view(B, K),  # ← 추가
        'size_2d': size_2d,                         # [B,K,2]
        'heading': heading_BKH,                     # [B,K,H,24]
        'size3d': size3d_BKH,                       # [B,K,H,3]
        'offset3d': offset3d_BKH,                   # [B,K,H,2]
        'depth': depth_BKH,                         # [B,K,H,2] (mu,logvar)
        'iou_conf': iou_conf_BKH                    # [B,K,H] or None
    }

    if to_numpy:
        for k, v in dets_pack.items():
            if v is None: continue
            dets_pack[k] = v.detach().cpu().numpy()
    return dets_pack


############### auxiliary function ############
def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim  = feat.size(2)  # get channel dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()   # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))   # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)     # B * len(ind) * C
    return feat


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)



if __name__ == '__main__':
    ## testing
    from lib.datasets.kitti import KITTI
    from torch.utils.data import DataLoader

    dataset = KITTI('../../data', 'train')
    dataloader = DataLoader(dataset=dataset, batch_size=2)
