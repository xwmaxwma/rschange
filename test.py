import torch
from torch import nn
from tqdm import tqdm
import prettytable
import time
import os
import multiprocessing.pool as mpp
import multiprocessing as mp

from train import *

import argparse
from utils.config import Config
from tools.mask_convert import mask_save

def get_args():
    parser = argparse.ArgumentParser('description=Change detection of remote sensing images')
    parser.add_argument("-c", "--config", type=str, default="configs/cdlama.py")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    cfg = Config.fromfile(args.config)

    ckpt = args.ckpt
    if ckpt is None:
        ckpt = cfg.test_ckpt_path
    assert ckpt is not None

    if args.output_dir:
        base_dir = args.output_dir
    else:
        base_dir = os.path.dirname(ckpt)
    masks_output_dir = os.path.join(base_dir, "mask_rgb") 

    model = myTrain.load_from_checkpoint(ckpt, map_location={'cuda:1':'cuda:0'}, cfg = cfg)
    model = model.to('cuda')

    model.eval()

    metric_cfg_1 = cfg.metric_cfg1
    metric_cfg_2 = cfg.metric_cfg2
    
    test_oa=torchmetrics.Accuracy(**metric_cfg_1).to('cuda')
    test_prec = torchmetrics.Precision(**metric_cfg_2).to('cuda')
    test_recall = torchmetrics.Recall(**metric_cfg_2).to('cuda')
    test_f1 = torchmetrics.F1Score(**metric_cfg_2).to('cuda')
    test_iou=torchmetrics.JaccardIndex(**metric_cfg_2).to('cuda')

    results = []
    with torch.no_grad():
        test_loader = build_dataloader(cfg.dataset_config, mode='test')
        for input in tqdm(test_loader):

            raw_predictions, mask, img_id = model(input[0].cuda(), input[1].cuda()), input[2].cuda(), input[3]

            if cfg.net == 'SARASNet':
                mask = Variable(resize_label(mask.data.cpu().numpy(), \
                                        size=raw_predictions.data.cpu().numpy().shape[2:]).to('cuda')).long()
                param = 1  # This parameter is balance precision and recall to get higher F1-score
                raw_predictions[:,1,:,:] = raw_predictions[:,1,:,:] + param 
                
            if cfg.argmax:
                pred = raw_predictions.argmax(dim=1)
            else:
                if cfg.net == 'maskcd':
                    pred = raw_predictions[0]
                    pred = pred > 0.5
                    pred.squeeze_(1)
                else:
                    pred = raw_predictions.squeeze(1)
                    pred = pred > 0.5

            test_oa(pred, mask)
            test_iou(pred, mask)
            test_prec(pred, mask)
            test_f1(pred, mask)
            test_recall(pred, mask)

            for i in range(raw_predictions.shape[0]):
                mask_real = mask[i].cpu().numpy()
                mask_pred = pred[i].cpu().numpy()
                mask_name = str(img_id[i])
                results.append((mask_real, mask_pred, masks_output_dir, mask_name))

    metrics = [test_prec.compute(),
               test_recall.compute(),
               test_f1.compute(),
               test_iou.compute()]
    
    total_metrics = [test_oa.compute().cpu().numpy(),
                     np.mean([item.cpu() for item in metrics[0]]),
                     np.mean([item.cpu() for item in metrics[1]]),
                     np.mean([item.cpu() for item in metrics[2]]),
                     np.mean([item.cpu() for item in metrics[3]])]

    result_table = prettytable.PrettyTable()
    result_table.field_names = ['Class', 'OA', 'Precision', 'Recall', 'F1_Score', 'IOU']

    for i in range(2):
        item = [i, '--']
        for j in range(len(metrics)):
            item.append(np.round(metrics[j][i].cpu().numpy(), 4))
        result_table.add_row(item)

    total = [np.round(v, 4) for v in total_metrics]
    total.insert(0, 'total')
    result_table.add_row(total)

    print(result_table)

    file_name = os.path.join(base_dir, "test_res.txt") 
    f = open(file_name,"a")
    current_time = time.strftime('%Y_%m_%d %H:%M:%S {}'.format(cfg.net),time.localtime(time.time()))
    f.write(current_time+'\n')
    f.write(str(result_table)+'\n')
 
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
    print(masks_output_dir)

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(mask_save, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))
