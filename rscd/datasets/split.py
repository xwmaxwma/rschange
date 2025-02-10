#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import random
from shutil import copyfile
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import argparse

parser = argparse.ArgumentParser(description='Splitting the Images')

parser.add_argument('--src', default='E:/zjuse/2308CD/rschangedetection/data/DSIFN', type=str,
                    help='path for the original dataset')
parser.add_argument('--tar', default='E:/zjuse/2308CD/rschangedetection/data/DSIFN/split_file', type=str,
                    help='path for saving the new dataset')
parser.add_argument('--image_sub_folder', default='t1,t2,mask', type=str,
                    help='name of subfolder inside the training, validation and test folders')
parser.add_argument('--set', default="train,val,test", type=str, help='evaluation mode')
parser.add_argument('--patch_width', default=256, type=int, help='Width of the cropped image patch')
parser.add_argument('--patch_height', default=256, type=int, help='Height of the cropped image patch')
parser.add_argument('--stride', default=256, type=int, help='Overlap area')

args = parser.parse_args()

src = args.src
tar = args.tar
modes1 = args.set.split(',')  # train, val
modes2 = args.image_sub_folder.split(',')  # t1,t2,mask
patch_H, patch_W = args.patch_width, args.patch_height  # image patch width and height
stride = args.stride  # overlap area

txt_list = dict()
txt_name = args.set + ".txt"
txt_path = tar + "/" + txt_name

# random_train_val = np.array(['train' for i in range()])



for mode1 in modes1:
    for mode2 in modes2:
        src_path = src + "/" + mode1 + "/" + mode2
        tar_path = tar + "/" + mode1 + "/" + mode2

        txt_list[mode2] = []

        os.makedirs(tar_path, exist_ok=True)
        files = os.listdir(src_path)
        # files = [x for x in os.listdir(src_path) if x.endswith(".jpg")]
        for file_ in files:
            full_filename = src_path + '/' + file_
            img = Image.open(full_filename)
            img_H, img_W = img.size

            if img_H >= patch_H and img_W >= patch_W:
                for x in range(0, img_H, stride):
                    x_str = x
                    x_end = x + patch_H
                    if x_str >= img_H:
                        break
                    for y in range(0, img_W, stride):
                        y_str = y
                        y_end = y + patch_W
                        if y_str >= img_W:
                            break
                        patch = img.crop((x_str, y_str, x_end, y_end))
                        image = file_[:-4] + '_' + str(x_str) + '_' + str(x_end) + '_' + str(y_str) + '_' + str(
                            y_end) + '.png'
                        save_path_image = tar_path + '/' + image
                        patch = np.array(patch, dtype=np.uint8)
                        patch = Image.fromarray(patch)
                        patch.save(save_path_image)
                        txt_list[mode2].append(save_path_image)

# for mode1 in modes1:
#     for mode2 in modes2:
#         src_path = src  + "/" + mode1 + "/" + mode2
#         tar_path = tar  + "/" + mode1 + "/" + mode2

#         txt_list[mode2] = []

#         os.makedirs(tar_path, exist_ok=True)
#         files = os.listdir(src_path)
#         # files = [x for x in os.listdir(src_path) if x.endswith(".jpg")]
#         for file_ in files:
#             full_filename = src_path + '/' + file_
#             img = cv2.imread(full_filename)
#             img_H, img_W, _ = img.shape
#             X = np.zeros_like(img, dtype=float)

#             if img_H > patch_H and img_W > patch_W:
#                 for x in range(0, img_W, patch_W-overlap):
#                     x_str = x
#                     x_end = x + patch_W
#                     if x_end > img_W:
#                         break
#                     for y in range(0, img_H, patch_H-overlap):
#                         y_str = y
#                         y_end = y + patch_H
#                         if y_end > img_H:
#                             break
#                         patch = img[y_str:y_end,x_str:x_end,:]
#                         image = file_[:-4]+'_'+str(y_str)+'_'+str(y_end)+'_'+str(x_str)+'_'+str(x_end)+'.png'
#                         save_path_image = tar_path + '/' + image
#                         cv2.imwrite(save_path_image, patch)
#                         txt_list[mode2].append(save_path_image)

with open(txt_path, 'w') as f:
    txt_1, txt_2, txt_3 = txt_list['A'], txt_list['B'], txt_list['label']
    for i in range(len(txt_1)):
        f.write(txt_1[i])
        f.write(" ")
        f.write(txt_2[i])
        f.write(" ")
        f.write(txt_3[i])
        f.write("\n")
    f.close()

# def tif2jpg(path):
#     tif_list = [x for x in os.listdir(path) if x.endswith(".tif")]
#     for num,i in enumerate(tif_list):
#         # img = plt.imread(path+'/'+i)
#         # img = Image.fromarray(img).convert("RGB")
#         # img.save(path+'/'+i.split('.')[0]+".jpg")
#         img = Image.open(path+'/'+i)
#         rgbimg = img.convert("L")
#         rgbimg.save(path+'/'+i.split('.')[0]+".jpg")
#         # img = cv2.imread(path+'/'+i, cv2.IMREAD_UNCHANGED)
#         # rgb_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#         # cv2.imwrite(path+'/'+i.split('.')[0]+".jpg", rgb_image)

# if __name__ == "__main__":
#     import cv2
#     from PIL import Image
#     import matplotlib.pyplot as plt
#     tif2jpg("/home/ma-user/work/xwma/data/DSIFN/test/mask")
