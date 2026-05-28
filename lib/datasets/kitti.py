import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.utils import get_angle_from_box3d,check_range
from lib.datasets.kitti_utils import get_objects_from_label
from lib.datasets.kitti_utils import Calibration
from lib.datasets.kitti_utils import get_affine_transform
from lib.datasets.kitti_utils import affine_transform
from lib.datasets.kitti_utils import compute_box_3d

import cv2 as cv
import torchvision.ops.roi_align as roi_align
import math
import random
from lib.datasets.kitti_utils import Object3d



class KITTI(data.Dataset):
    def __init__(self, root_dir, split, cfg):
        # basic configuration
        self.num_classes = 3
        self.max_objs = 50
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        self.resolution = np.array([1280, 384])  # W * H
        self.use_3d_center = cfg['use_3d_center']
        self.writelist = cfg['writelist']
        if cfg['class_merging']:
            self.writelist.extend(['Van', 'Truck'])
        if cfg['use_dontcare']:
            self.writelist.extend(['DontCare'])
        '''    
        ['Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
         'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
         'Cyclist': np.array([1.76282397,0.59706367,1.73698127])] 
        ''' 
        ##l,w,h
        self.cls_mean_size = np.array([[1.76255119    ,0.66068622   , 0.84422524   ],
                                       [1.52563191462 ,1.62856739989, 3.88311640418],
                                       [1.73698127    ,0.59706367   , 1.76282397   ]])                              
                              
        # data split loading
        assert split in ['train', 'val', 'trainval', 'test']
        self.split = split
        split_dir = os.path.join(root_dir, cfg['data_dir'], 'ImageSets', split + '.txt')
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        # path configuration
        self.data_dir = os.path.join(root_dir, cfg['data_dir'], 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.depth_dir = os.path.join(self.data_dir, 'depth')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')
        
        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False  # 'cal'
        self.divalign_aug_strength = cfg.get('divalign_aug_strength', 'mild') 
        self.random_flip = cfg['random_flip']
        self.random_crop = cfg['random_crop']
        self.scale = cfg['scale']
        self.shift = cfg['shift']

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # others
        self.downsample = 4

    # ---------- helpers ----------
    def _same_calib(self, c1, c2, eps=1e-6):
        return (abs(c1.cu - c2.cu) < eps and
                abs(c1.cv - c2.cv) < eps and
                abs(c1.fu - c2.fu) < eps and
                abs(c1.fv - c2.fv) < eps)

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file).convert('RGB')    # (H, W, 3) RGB mode


    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)
    
    def __len__(self):
        return self.idx_list.__len__()

        # ======================= Aug Utils (in-class) =======================
    def _pil_to_uint8(self, img_pil):
        arr = np.array(img_pil)
        if arr.dtype != np.uint8:
            if arr.dtype == np.float32:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        return arr

    def _uint8_to_pil(self, arr):
        return Image.fromarray(arr.astype(np.uint8))

    # ---------- Blur ----------
    def _aug_gaussian_blur(self, img, k=3, sigma=0):
        k = k if k % 2 == 1 else k+1
        return cv.GaussianBlur(img, (k, k), sigmaX=sigma)

    def _aug_motion_blur(self, img, k=5):
        k = max(3, int(k) | 1)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k//2, :] = 1.0 / k
        return cv.filter2D(img, -1, kernel)

    def _aug_defocus_blur(self, img, radius=2):
        r = max(1, int(radius))
        k = 2*r + 1
        y, x = np.ogrid[-r:r+1, -r:r+1]
        mask = x*x + y*y <= r*r
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[mask] = 1.0
        kernel /= kernel.sum()
        return cv.filter2D(img, -1, kernel)

    # ---------- Noise ----------
    def _aug_gaussian_noise(self, img, std=8):
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        out = img.astype(np.float32) + noise
        return np.clip(out, 0, 255).astype(np.uint8)

    def _aug_shot_noise(self, img, scale=0.05):
        x = img.astype(np.float32) / 255.0
        x = np.clip(x, 0.0, 1.0)
        lam = np.maximum(x / scale, 1e-6)
        y = np.random.poisson(lam).astype(np.float32) * scale
        return np.clip(y * 255.0, 0, 255).astype(np.uint8)

    def _aug_impulse_noise(self, img, p=0.004):
        out = img.copy()
        n = int(p * img.shape[0] * img.shape[1])
        if n <= 0: 
            return out
        coords = (np.random.randint(0, img.shape[0], n), np.random.randint(0, img.shape[1], n))
        out[coords] = 255
        coords = (np.random.randint(0, img.shape[0], n), np.random.randint(0, img.shape[1], n))
        out[coords] = 0
        return out

    def _aug_speckle_noise(self, img, std=0.02):
        noise = np.random.randn(*img.shape).astype(np.float32) * std
        out = img.astype(np.float32) + img.astype(np.float32) * noise
        return np.clip(out, 0, 255).astype(np.uint8)

    # ---------- Digital ----------
    def _aug_brightness(self, img, delta=0.08):
        factor = 1.0 + np.random.uniform(-delta, delta)
        out = img.astype(np.float32) * factor
        return np.clip(out, 0, 255).astype(np.uint8)

    def _aug_contrast(self, img, delta=0.08):
        factor = 1.0 + np.random.uniform(-delta, delta)
        mean = np.mean(img, axis=(0,1), keepdims=True)
        out = (img.astype(np.float32) - mean) * factor + mean
        return np.clip(out, 0, 255).astype(np.uint8)

    def _aug_saturation(self, img, delta=0.08):
        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV).astype(np.float32)
        factor = 1.0 + np.random.uniform(-delta, delta)
        hsv[...,1] = np.clip(hsv[...,1] * factor, 0, 255)
        out = cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2RGB)
        return out

    def _aug_jpeg(self, img, quality=85):
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, enc = cv.imencode('.jpg', cv.cvtColor(img, cv.COLOR_RGB2BGR), encode_param)
        if not ok: 
            return img
        dec = cv.imdecode(enc, cv.IMREAD_COLOR)
        return cv.cvtColor(dec, cv.COLOR_BGR2RGB)

    def _aug_fourier_phase_scale(self, img, phase_scale=0.9):
        img_f = img.astype(np.float32) / 255.0
        out = np.zeros_like(img_f)
        for c in range(3):
            F = np.fft.fft2(img_f[..., c])
            A, P = np.abs(F), np.angle(F)
            P2 = P * phase_scale
            F2 = A * np.exp(1j * P2)
            ch = np.fft.ifft2(F2).real
            out[..., c] = np.clip(ch, 0.0, 1.0)
        return (out * 255.0).astype(np.uint8)

    def _aug_fourier_highpass(self, img, cutoff=3):
        img_f = img.astype(np.float32) / 255.0
        H, W = img.shape[:2]
        out = np.zeros_like(img_f)
        for c in range(3):
            F = np.fft.fftshift(np.fft.fft2(img_f[..., c]))
            cy, cx = H // 2, W // 2
            r = int(cutoff)
            F[cy-r:cy+r+1, cx-r:cx+r+1] = 0
            ch = np.fft.ifft2(np.fft.ifftshift(F)).real
            out[..., c] = np.clip(ch, 0.0, 1.0)
        return (out * 255.0).astype(np.uint8)

    def _apply_divalign_aug(self, img_pil, strength=None):
        """
        img_pil: PIL RGB
        strength: 'mild' | 'medium'
        """
        if strength is None:
            strength = self.divalign_aug_strength

        arr = self._pil_to_uint8(img_pil)

        if strength == 'mild':
            blur_k   = np.random.choice([3,5])
            motion_k = 5
            def_r    = 2
            gstd     = np.random.choice([6,8])
            speckle  = 0.02
            imp_p    = 0.004
            bright   = 0.08
            contr    = 0.08
            sat      = 0.08
            jpeg_q   = np.random.choice([75,90])
            phase_s  = 0.90
            hp_cut   = 3
        else:  # 'medium'
            blur_k   = np.random.choice([5,7])
            motion_k = np.random.choice([7,9])
            def_r    = 3
            gstd     = np.random.choice([8,10])
            speckle  = 0.03
            imp_p    = 0.006
            bright   = 0.12
            contr    = 0.12
            sat      = 0.12
            jpeg_q   = np.random.choice([65,85])
            phase_s  = 0.85
            hp_cut   = 4

        ops = [
            lambda x: self._aug_gaussian_blur(x, k=blur_k),
            lambda x: self._aug_motion_blur(x, k=motion_k),
            lambda x: self._aug_defocus_blur(x, radius=def_r),
            lambda x: self._aug_gaussian_noise(x, std=gstd),
            lambda x: self._aug_shot_noise(x, scale=0.02),
            lambda x: self._aug_impulse_noise(x, p=imp_p),
            lambda x: self._aug_speckle_noise(x, std=speckle),
            lambda x: self._aug_brightness(x, delta=bright),
            lambda x: self._aug_contrast(x,  delta=contr),
            lambda x: self._aug_saturation(x,delta=sat),
            lambda x: self._aug_jpeg(x, quality=jpeg_q),
            lambda x: self._aug_fourier_phase_scale(x, phase_scale=phase_s),
            lambda x: self._aug_fourier_highpass(x, cutoff=hp_cut),
        ]

        
        out = np.random.choice(ops)(arr)

        if np.random.rand() < 0.15:
            ops2 = ops[3:]  # photometric
            out = np.random.choice(ops2)(out)

        return self._uint8_to_pil(out)




    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        img = self.get_image(index)
        img_size = np.array(img.size)
        if self.split!='test':
            dst_W, dst_H = img_size

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size = img_size
        random_crop_flag, random_flip_flag = False, False
        random_mix_flag, divalign_aug_enable = False, False
        calib = self.get_calib(index)

        if self.data_augmentation:
            if np.random.random() < 0.5:
                random_mix_flag = True
            if np.random.random() < 0.5:
                divalign_aug_enable = True
            
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random() < self.random_crop:
                crop_size = img_size * np.clip(np.random.randn()*self.scale + 1, 1 - self.scale, 1 + self.scale)
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                
        if random_mix_flag:
            count_num = 0
            random_mix_flag = False
            while count_num < 50:
                count_num += 1
                random_index = np.random.randint(len(self.idx_list))
                random_index = int(self.idx_list[random_index])
                calib_temp = self.get_calib(random_index)
                
                if self._same_calib(calib_temp, calib):
                    img_temp = self.get_image(random_index)
                    img_size_temp = np.array(img_temp.size)  
                    dst_W_temp, dst_H_temp = img_size_temp
                    if dst_W_temp == dst_W and dst_H_temp == dst_H:
                        objects_1 = self.get_label(index)
                        objects_2 = self.get_label(random_index)
                        if len(objects_1) + len(objects_2) < self.max_objs: 
                            random_mix_flag = True
                            if random_flip_flag == True:
                                img_temp = img_temp.transpose(Image.FLIP_LEFT_RIGHT)
                            # alpha-random mixup
                            alpha = float(np.random.uniform(0.3, 0.7))
                            img_blend = Image.blend(img, img_temp, alpha=alpha)
                            img = img_blend
                            break
        
        if divalign_aug_enable:
            img = self._apply_divalign_aug(img)  

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        
        coord_range = np.array([center-crop_size/2,center+crop_size/2]).astype(np.float32)
        
        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W
        
        features_size = self.resolution // self.downsample  # W * H

        #  ============================   get labels   ==============================
        if self.split!='test':
            objects = self.get_label(index)
            # data augmentation for labels
            if random_flip_flag:
                calib.flip(img_size)
                for object in objects:
                    [x1, _, x2, _] = object.box2d
                    object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                    object.ry = np.pi - object.ry
                    object.pos[0] *= -1
                    if object.ry > np.pi:  object.ry -= 2 * np.pi
                    if object.ry < -np.pi: object.ry += 2 * np.pi
            # labels encoding
            heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32) # C * H * W
            size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            depth = np.zeros((self.max_objs, 1), dtype=np.float32)
            heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
            heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
            src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
            height2d = np.zeros((self.max_objs, 1), dtype=np.float32)
            cls_ids = np.zeros((self.max_objs), dtype=np.int64)
            indices = np.zeros((self.max_objs), dtype=np.int64)
            mask_2d = np.zeros((self.max_objs), dtype=np.bool_)  # FIX: np.bool -> np.bool_
            object_num = len(objects) if len(objects) < self.max_objs else self.max_objs

            vis_depth = np.zeros((self.max_objs, 7, 7), dtype=np.float32)

            count = 0
            for i in range(object_num):
                
                # filter objects by writelist
                if objects[i].cls_type not in self.writelist:
                    continue

                # process 2d bbox & get 2d center
                bbox_2d = objects[i].box2d.copy()
                # add affine transformation for 2d boxes.
                bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
                # modify the 2d bbox according to pre-compute downsample ratio
                bbox_2d[:] /= self.downsample

                # process 3d bbox & get 3d center
                center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
                center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                center_3d = center_3d[0]  # shape adjustment
                center_3d = affine_transform(center_3d.reshape(-1), trans)
                center_3d /= self.downsample

                # generate the center of gaussian heatmap [optional: 3d center or 2d center]
                center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)

                if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: 
                    continue

                if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: 
                    continue
    
                # generate the radius of gaussian heatmap
                w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                radius = gaussian_radius((w, h))
                radius = max(0, int(radius))
    
                if objects[i].cls_type in ['Van', 'Truck', 'DontCare']:
                    draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                    continue
    
                cls_id = self.cls2id[objects[i].cls_type]
                cls_ids[i] = cls_id
                draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)
    
                # encoding 2d/3d offset & 2d size
                indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
                offset_2d[i] = center_2d - center_heatmap
                size_2d[i] = 1. * w, 1. * h
    
                # encoding depth
                depth[i] = objects[i].pos[-1]
    
                # encoding heading angle
                heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0]+objects[i].box2d[2])/2)
                if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                if heading_angle < -np.pi: heading_angle += 2 * np.pi
                heading_bin[i], heading_res[i] = angle2class(heading_angle)

                offset_3d[i] = center_3d - center_heatmap
                src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
                size_3d[i] = src_size_3d[i] - mean_size

                if objects[i].trucation <=0.5 and objects[i].occlusion<=2:
                    mask_2d[i] = 1

                vis_depth[i] = depth[i]
            if random_mix_flag == True:
                objects = self.get_label(random_index)
                # data augmentation for labels
                if random_flip_flag:
                    for object in objects:
                        [x1, _, x2, _] = object.box2d
                        object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                        object.ry = np.pi - object.ry
                        object.pos[0] *= -1
                        if object.ry > np.pi:  object.ry -= 2 * np.pi
                        if object.ry < -np.pi: object.ry += 2 * np.pi
                object_num_temp = len(objects) if len(objects) < (self.max_objs - object_num) else (self.max_objs - object_num)
                for i in range(object_num_temp):
                    if objects[i].cls_type not in self.writelist:
                        continue

                    if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                        continue
                    # process 2d bbox & get 2d center
                    bbox_2d = objects[i].box2d.copy()
                    # add affine transformation for 2d boxes.
                    bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                    bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
                    # modify the 2d bbox according to pre-compute downsample ratio
                    bbox_2d[:] /= self.downsample

                    # process 3d bbox & get 3d center
                    center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
                    center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                    center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                    center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                    center_3d = center_3d[0]  # shape adjustment
                    center_3d = affine_transform(center_3d.reshape(-1), trans)
                    center_3d /= self.downsample

                    # generate the center of gaussian heatmap [optional: 3d center or 2d center]
                    center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
                    if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
                    if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue
        
                    # generate the radius of gaussian heatmap
                    w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                    radius = gaussian_radius((w, h))
                    radius = max(0, int(radius))
        
                    if objects[i].cls_type in ['Van', 'Truck', 'DontCare']:
                        draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                        continue
        
                    cls_id = self.cls2id[objects[i].cls_type]
                    cls_ids[i + object_num] = cls_id
                    draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)
        
                    # encoding 2d/3d offset & 2d size
                    indices[i + object_num] = center_heatmap[1] * features_size[0] + center_heatmap[0]
                    offset_2d[i + object_num] = center_2d - center_heatmap
                    size_2d[i + object_num] = 1. * w, 1. * h
        
                    # encoding depth
                    depth[i + object_num] = objects[i].pos[-1]
        
                    # encoding heading angle
                    heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0]+objects[i].box2d[2])/2)
                    if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                    if heading_angle < -np.pi: heading_angle += 2 * np.pi
                    heading_bin[i + object_num], heading_res[i + object_num] = angle2class(heading_angle)

                    offset_3d[i + object_num] = center_3d - center_heatmap
                    src_size_3d[i + object_num] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                    mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
                    size_3d[i + object_num] = src_size_3d[i + object_num] - mean_size

                    if objects[i].trucation <=0.5 and objects[i].occlusion<=2:
                        mask_2d[i + object_num] = 1

                    vis_depth[i + object_num] = depth[i + object_num]

            targets = {'depth': depth,
                       'size_2d': size_2d,
                       'heatmap': heatmap,
                       'offset_2d': offset_2d,
                       'indices': indices,
                       'size_3d': size_3d,
                       'offset_3d': offset_3d,
                       'heading_bin': heading_bin,
                       'heading_res': heading_res,
                       'cls_ids': cls_ids,
                       'mask_2d': mask_2d,
                       'vis_depth': vis_depth,
                       }
        else:
            targets = {}

        inputs = img
        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size/features_size}

        return inputs, calib.P2, coord_range, targets, info   #calib.P2


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    cfg = {'random_flip':0.0, 'random_crop':1.0, 'scale':0.4, 'shift':0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist':['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center':False,
           'data_dir': 'kitti'}  
    dataset = KITTI('../../data', 'train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs, P2, coord_range, targets, info) in enumerate(dataloader): 
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img.show()

        # test heatmap
        heatmap = targets['heatmap'][0]  # image id
        heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        heatmap.show()
        break

    # print ground truth first
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
