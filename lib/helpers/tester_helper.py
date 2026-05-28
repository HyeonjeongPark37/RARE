import os
import tqdm

import torch
import numpy as np

from tools import eval

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections

class Tester(object):
    def __init__(self, cfg, model, data_loader, logger):
        self.cfg = cfg
        self.cfg_tester = cfg['tester']
        self.model = model
        self.data_loader = data_loader
        self.logger = logger
        self.class_name = data_loader.dataset.class_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.eval_cls = cfg['dataset']['eval_cls']
        self.eval_dataset = cfg['dataset']['eval_dataset'] if 'eval_dataset' in cfg['dataset'].keys() else cfg['dataset']['type']
        self.tester_metrics = 'kitti' if 'tester_metrics' not in self.cfg['tester'].keys() else self.cfg['tester']['tester_metrics']
        self.H = None
        
        if self.cfg['tester'].get('resume_model', None):
            load_checkpoint(model = self.model,
                        optimizer = None,
                        filename = cfg['tester']['resume_model'],
                        logger = self.logger,
                        map_location=self.device)

        self.model = torch.nn.DataParallel(model).to(self.device)

    def test(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        all_meta = {}                   

        progress_bar = tqdm.tqdm(total=len(self.data_loader), leave=True, desc='Evaluation Progress')
        for batch_idx, (inputs, calibs, coord_ranges, _, info) in enumerate(self.data_loader):
            # load evaluation data and move data to current device.
            if type(inputs) != dict:
                inputs = inputs.to(self.device)
            else:
                for key in inputs.keys(): inputs[key] = inputs[key].to(self.device)
            calibs = calibs.to(self.device)
            coord_ranges = coord_ranges.to(self.device)

            # the outputs of centernet
            outputs = self.model(inputs, coord_ranges, calibs, K=50, mode='test')
            dets_pack = extract_dets_from_outputs(outputs=outputs, K=50, to_numpy=True)
            if self.H is None:
                self.H = int(dets_pack['depth'].shape[2])
                    
            # python-side calib list & numpy info
            calibs_py = [self.data_loader.dataset.get_calib(index) for index in info['img_id']]
            info_np = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.data_loader.dataset.cls_mean_size

            dets =  decode_detections(
                dets_pack, info_np, calibs_py, cls_mean_size, 
                threshold=self.cfg['tester']['threshold'],
                use_iou_conf=True,)

            results.update(dets)
            progress_bar.update()

        # save the result for evaluation.
        output_dir = self.cfg_tester['out_dir']
        self.save_results(results, output_dir)
        progress_bar.close()
        
        if self.eval_dataset == "kitti":
            gt_folder = '/your_label_path/KITTIDataset/training/label_2'
        else:
            raise NotImplementedError

        # Now run evaluation code
        if self.tester_metrics == 'kitti':
            eval_res = eval.eval_from_scrach(
                    gt_folder,
                    output_dir,
                    '',
                    self.eval_cls,
                    ap_mode=40
                )
        else:
            print("Not suppored dataset for evaluation")
            
    def save_results(self, results, output_dir='./outputs'):
        os.makedirs(output_dir, exist_ok=True)
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

def _wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def _circ_mean(angles, weights):
    s = np.sum(weights * np.sin(angles))
    c = np.sum(weights * np.cos(angles))
    if s == 0.0 and c == 0.0:
        return float(angles[0])
    return float(np.arctan2(s, c))


def _normalize_weights(w, mode='sum', temp=None, power=1.0):
    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, 0.0)
    if power != 1.0:
        w = np.power(w, power)

    if (mode == 'softmax') or (temp is not None):
        t = 1.0 if temp is None else float(temp)
        w = np.exp((w - np.max(w)) / max(1e-6, t))
        s = np.sum(w)
        return w / s if s > 0 else np.ones_like(w) / len(w)
    else:
        s = np.sum(w)
        return w / s if s > 0 else np.ones_like(w) / len(w)

