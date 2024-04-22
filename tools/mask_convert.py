import numpy as np
import argparse
import glob
import os
import sys
import torch
import cv2
import random
import time
import multiprocessing.pool as mpp
import multiprocessing as mp
SEED = 66

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
def label2rgb(mask, mask_pred):
    real_1 = (mask == 1)
    real_0 = (mask == 0)
    pred_1 = (mask_pred == 1)
    pred_0 = (mask_pred == 0)

    TP = np.logical_and(real_1, pred_1)
    TN = np.logical_and(real_0, pred_0)
    FN = np.logical_and(real_1, pred_0)
    FP = np.logical_and(real_0, pred_1)

    mask_TP = TP[np.newaxis, :, :]
    mask_TN = TN[np.newaxis, :, :]
    mask_FN = FN[np.newaxis, :, :]
    mask_FP = FP[np.newaxis, :, :]

    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_rgb[np.all(mask_TP, axis=0)] = [255, 255, 255] # TP
    mask_rgb[np.all(mask_TN, axis=0)] = [0, 0, 0] # TN
    mask_rgb[np.all(mask_FN, axis=0)] = [0, 255, 0] # FN
    mask_rgb[np.all(mask_FP, axis=0)] = [255, 0, 0] # FP

    return mask_rgb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Vaihingen")
    parser.add_argument("--mask-dir", default="data/Test/masks")
    parser.add_argument("--output-mask-dir", default="data/Test/masks_rgb")
    return parser.parse_args()

def mask_save(inp):
    (mask, mask_pred, masks_output_dir, file_name) = inp
    out_mask_path = os.path.join(masks_output_dir, "{}.png".format(file_name))

    label = label2rgb(mask.copy(), mask_pred.copy())

    rgb_label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    cv2.imwrite(out_mask_path, rgb_label)

# def get_rgb(inp):
#     (mask_path, masks_output_dir,dataset) = inp
#     mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
#     mask_bgr = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
#     mask = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
#     if dataset == "LoveDA":
#         rgb_label = loveda_label2rgb(mask.copy())
#     elif dataset == "Vaihingen":
#         rgb_label = vaihingen_label2rgb(mask.copy())
#     elif dataset == "Potsdam":
#         rgb_label = potsdam_label2rgb(mask.copy())
#     elif dataset == "uavid":
#         rgb_label = uavid_label2rgb(mask.copy())
#     else: return
#     #rgb_label = cv2.cvtColor(rgb_label, cv2.COLOR_RGB2BGR)

#     out_mask_path_rgb = os.path.join(masks_output_dir, "{}.png".format(mask_filename))
#     rgb_label = cv2.cvtColor(rgb_label, cv2.COLOR_BGR2RGB)
#     cv2.imwrite(out_mask_path_rgb, rgb_label)

# if __name__ == '__main__':
#     base_path = "/home/xwma/lrr/rssegmentation/"
#     args = parse_args()
#     dataset = args.dataset

#     seed_everything(SEED)
#     masks_dir = args.mask_dir
#     masks_output_dir = args.output_mask_dir
#     masks_dir = base_path + masks_dir
#     masks_output_dir = base_path + masks_output_dir

#     mask_paths = glob.glob(os.path.join(masks_dir, "*.png"))
#     inp = [(mask_path, masks_output_dir, dataset) for mask_path in mask_paths]
#     if not os.path.exists(masks_output_dir):
#         os.makedirs(masks_output_dir)

#     t0 = time.time()
#     mpp.Pool(processes=mp.cpu_count()).map(get_rgb, inp)
#     t1 = time.time()
#     split_time = t1 - t0
#     print('images spliting spends: {} s'.format(split_time))
