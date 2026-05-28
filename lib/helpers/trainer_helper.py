import os
import tqdm
import wandb

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from lib.helpers.save_helper import get_checkpoint_state, save_checkpoint, load_checkpoint
from lib.losses.loss_function import RARELoss, Hierarchical_Task_Learning
from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections

from tools import eval 


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 output_path):
        self.cfg   = cfg
        self.cfg_train = cfg['trainer']
        self.cfg_test = cfg['tester']
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_name = test_loader.dataset.class_name
        self.eval_cls = cfg['dataset']['eval_cls']
        self.eval_dataset =  cfg['dataset']['eval_dataset'] if 'eval_dataset' in cfg['dataset'].keys() else cfg['dataset']['type']
        self.tester_metrics = 'kitti' if 'tester_metrics' not in self.cfg_test.keys() else self.cfg_test['tester_metrics']

        self.val_best_m_result = -1
        self.val_best_m_epoch = -1
        self.train_best_m_result = -1
        self.train_best_m_epoch = -1
        
        self.best_h_indices = []
        self.depth_diff = []
        self.H = None

        self.output_path = output_path
        
        if self.cfg_train.get('resume_model', None):
            self.logger.info('Resume training: ', self.cfg_train['resume_model'])
            assert os.path.exists(self.cfg_train['resume_model'])
            self.epoch = load_checkpoint(self.model, self.optimizer, self.cfg_train['resume_model'], self.logger, map_location=self.device)
            self.lr_scheduler.last_epoch = self.epoch - 1

        self.model = torch.nn.DataParallel(model).to(self.device)

    def train(self):
        start_epoch = self.epoch
        ei_loss = self.compute_e0_loss()
        loss_weightor = Hierarchical_Task_Learning(ei_loss)
        for epoch in range(start_epoch, self.cfg_train['max_epoch']):
            # train one epoch
            self.logger.info('------ TRAIN EPOCH %03d ------' %(epoch + 1))
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.logger.info('Learning Rate: %f' % self.warmup_lr_scheduler.get_lr()[0])
            else:
                self.logger.info('Learning Rate: %f' % self.lr_scheduler.get_lr()[0])

            # reset numpy seed.
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            loss_weights = loss_weightor.compute_weight(ei_loss,self.epoch)

            log_str = 'Weights: '
            for key in sorted(loss_weights.keys()):
                log_str += ' %s:%.4f,' %(key[:-4], loss_weights[key])   
            self.logger.info(log_str)

            ei_loss = self.train_one_epoch(loss_weights)
            self.epoch += 1
            
            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            if ((self.epoch % self.cfg_train['eval_frequency']) == 0 and \
                self.epoch >= self.cfg_train['eval_start']):
            
                self.logger.info('------ EVAL EPOCH %03d ------' % (self.epoch))
                os.makedirs(self.output_path+'/checkpoints', exist_ok=True)
                
                if self.eval_dataset == 'kitti':
                    # car_train, car_val = self.eval_one_epoch(epoch)
                    car_val = self.eval_one_epoch(epoch)

                    def log_result(tag, result, best_result_attr, best_epoch_attr, ckpt_name_prefix=None):
                        e, m, h = result['3d@0.70']
                        wandb.log({
                            f'eval/{tag}_3D@0.7-easy': e,
                            f'eval/{tag}_3D@0.7-moderate': m,
                            f'eval/{tag}_3D@0.7-hard': h,
                            'eval/epoch': self.epoch
                        })
                        self.logger.info(f"{tag.capitalize()} Easy: {e}")
                        self.logger.info(f"{tag.capitalize()} Moderate: {m}")
                        self.logger.info(f"{tag.capitalize()} Hard: {h}")

                        if m > getattr(self, best_result_attr):
                            setattr(self, best_result_attr, m)
                            setattr(self, best_epoch_attr, self.epoch)
                            wandb.log({f'[BEST_{tag.upper()}] 3D@0.7-moderate': m})
                            self.logger.info(f"{tag.capitalize()} Best Moderate Updated: {m}, epoch: {self.epoch}")

                            if ckpt_name_prefix:
                                ckpt_name = os.path.join(self.output_path + '/checkpoints', f'{ckpt_name_prefix}_epoch_{self.epoch}')
                                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name, self.logger)

                    # Log all three evaluations
                    # log_result("train", car_train, "train_best_m_result", "train_best_m_epoch")
                    log_result("val", car_val, "val_best_m_result", "val_best_m_epoch", "checkpoint")
                
        if self.epoch == self.cfg_train['max_epoch']:
            ckpt_name = os.path.join(self.output_path+'/checkpoints', 'checkpoint_last_epoch_%d' % self.epoch)
            save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name, self.logger)
            
        return None
    
    def compute_e0_loss(self):
        self.model.train()
        disp_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=True, desc='pre-training loss stat')
        with torch.no_grad():        
            for batch_idx, (inputs, calibs,coord_ranges, targets, info) in enumerate(self.train_loader):
                if type(inputs) != dict:
                    inputs = inputs.to(self.device)
                else:
                    for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
                calibs = calibs.to(self.device)
                coord_ranges = coord_ranges.to(self.device)
                for key in targets.keys():
                    targets[key] = targets[key].to(self.device)
    
                # train one batch
                criterion = RARELoss(self.epoch, self.device)
                outputs = self.model(inputs,coord_ranges,calibs,targets)
                _, loss_terms = criterion(outputs, targets, calibs, coord_ranges)
                
                trained_batch = batch_idx + 1
                # accumulate statistics
                for key in loss_terms.keys():
                    if key not in disp_dict.keys():
                        disp_dict[key] = 0
                    disp_dict[key] += loss_terms[key]      
                progress_bar.update()
            progress_bar.close()
            for key in disp_dict.keys():
                disp_dict[key] /= trained_batch             
        return disp_dict

    def train_one_epoch(self,loss_weights=None):
        self.model.train()

        disp_dict = {}
        stat_dict = {}

        for batch_idx, (inputs, calibs,coord_ranges, targets, info) in enumerate(self.train_loader):
            
            if type(inputs) != dict:
                inputs = inputs.to(self.device)
            else:
                for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
            calibs = calibs.to(self.device)
            coord_ranges = coord_ranges.to(self.device)
            for key in targets.keys(): targets[key] = targets[key].to(self.device)
            
            # train one batch
            self.optimizer.zero_grad()
            criterion = RARELoss(self.epoch, self.device)
            outputs = self.model(inputs,coord_ranges,calibs,targets)

            total_loss, loss_terms = criterion(outputs, targets, calibs, coord_ranges)
            
            if loss_weights is not None:
                total_loss = torch.zeros(1).cuda()
                for key in loss_weights.keys():
                    total_loss += loss_weights[key].detach()*loss_terms[key]
            total_loss.backward()
            self.optimizer.step()

            trained_batch = batch_idx + 1

            # accumulate statistics
            for key in loss_terms.keys():
                if key not in stat_dict.keys():
                    stat_dict[key] = 0

                if isinstance(loss_terms[key], int):
                    stat_dict[key] += (loss_terms[key])
                else:
                    stat_dict[key] += (loss_terms[key]).detach()
            for key in loss_terms.keys():
                if key not in disp_dict.keys():
                    disp_dict[key] = 0
                # disp_dict[key] += loss_terms[key]
                if isinstance(loss_terms[key], int):
                    disp_dict[key] += (loss_terms[key])
                else:
                    disp_dict[key] += (loss_terms[key]).detach()
            
            # display statistics in terminal
            if trained_batch % self.cfg_train['disp_frequency'] == 0:
                freq = self.cfg_train['disp_frequency']
                log_str = 'BATCH[%04d/%04d]' % (trained_batch, len(self.train_loader))

                global_step = self.epoch * len(self.train_loader) + trained_batch

                for key in sorted(disp_dict.keys()):
                    val = disp_dict[key] / freq
                    if torch.is_tensor(val):
                        val = float(val.detach().cpu())
                    else:
                        val = float(val)
                    log_str += f' {key}:{val:.4f},'

                    wandb.log({key: val}, step=global_step)  # step 빼고 싶으면 인자 제거
                    disp_dict[key] = 0

                self.logger.info(log_str)
        
        for key in stat_dict.keys():
            stat_dict[key] /= trained_batch
                            
        return stat_dict    

    def eval_one_epoch(self, epoch):
        self.model.eval()

        if self.eval_dataset == "kitti":
            gt_folder = '/your_label_path/KITTIDataset/training/label_2'
        else:
            raise NotImplementedError

        def evaluate_loader(loader, vis_out_dir, save_folder, wandb_prefix):
            results = {}
            stat_dict, disp_dict = {}, {}

            progress_bar = tqdm.tqdm(total=len(loader), leave=True, desc=f'{wandb_prefix} Evaluation')
            with torch.no_grad():
                for batch_idx, (inputs, calibs, coord_ranges, targets, info) in enumerate(loader):
                    
                    if isinstance(inputs, dict):
                        for k in inputs: inputs[k] = inputs[k].to(self.device)
                    else:
                        inputs = inputs.to(self.device)
                    calibs, coord_ranges = calibs.to(self.device), coord_ranges.to(self.device)
                    for k in targets: targets[k] = targets[k].to(self.device)

                    # Loss & confidence computation
                    criterion = RARELoss(self.epoch, self.device)
                    outputs_for_logging = self.model(inputs, coord_ranges, calibs, targets)

                    total_loss, loss_terms = criterion(outputs_for_logging, targets, calibs, coord_ranges)
                    
                    # Loss logging
                    for key in loss_terms:
                        stat_dict[key] = stat_dict.get(key, 0) + loss_terms[key].detach()
                        disp_dict[key] = disp_dict.get(key, 0) + loss_terms[key].detach()

                    if (batch_idx + 1) % self.cfg_train['disp_frequency'] == 0:
                        log_str = f"{wandb_prefix} BATCH[{batch_idx+1:04d}/{len(loader):04d}]"
                        for key in sorted(disp_dict.keys()):
                            disp_dict[key] /= self.cfg_train['disp_frequency']
                            log_str += f" {key}:{disp_dict[key]:.4f},"
                            wandb.log({f"{wandb_prefix}/{key}": disp_dict[key]})
                            disp_dict[key] = 0
                        self.logger.info(log_str)

                    # Decode detections
                    outputs = self.model(inputs, coord_ranges, calibs, K=50, mode='val')
                    
                    dets_pack = extract_dets_from_outputs(outputs=outputs, K=50, to_numpy=True)
                    if self.H is None:
                        self.H = int(dets_pack['depth'].shape[2])
                    
                    calibs_np = [self.test_loader.dataset.get_calib(idx) for idx in info['img_id']]
                    info_np = {k: v.detach().cpu().numpy() for k, v in info.items()}
                    cls_mean_size = self.test_loader.dataset.cls_mean_size
        
                    dets = decode_detections(dets_pack = dets_pack,
                                     info = info_np,
                                     calibs = calibs_np,
                                     cls_mean_size=cls_mean_size,
                                     threshold = self.cfg['tester']['threshold'])

                    results.update(dets)
                    progress_bar.update()
                
                progress_bar.close()
            
                
            os.makedirs(save_folder, exist_ok=True)
            self.save_results(results, save_folder)

            if self.tester_metrics == 'kitti':
                eval_res = eval.eval_from_scrach(
                    gt_folder,
                    save_folder,
                    os.path.join(vis_out_dir, f'eval_{wandb_prefix}_epoch{self.epoch}.png'),
                    self.eval_cls,
                    ap_mode=40
                )
                return eval_res    
            else:
                return None

        if self.tester_metrics == 'kitti':
            # res_train = evaluate_loader(self.train_loader, self.train_vis_out_dir, self.output_path + "/train_data", "train_set")
            res_test = evaluate_loader(self.test_loader, self.val_vis_out_dir, self.output_path + "/val_data", "val_set")
            # return  res_train, res_test
            return res_test      
        else:
            print("Done")
            return None

    def save_results(self, results, output_dir='./outputs'):
        for img_id in results.keys():
            out_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            f = open(out_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()        
        
