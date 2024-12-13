import os
import sys
sys.path.append('.')

import matplotlib.pyplot as plt
from utilss import GradCAM, show_cam_on_image, center_crop_img
import math
import argparse
from utils.config import Config
from train import *

def get_args():
    # input x: B, L, C
    # if not, please adjust the order
    parser = argparse.ArgumentParser('description=Change detection of remote sensing images')
    parser.add_argument("-c", "--config", type=str, default="configs/cdmask.py")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--imgsize", default=256)
    return parser.parse_args()

class ResizeTransform:
    def __init__(self, im_h: int, im_w: int):
        self.height = im_h
        self.width = im_w

    def __call__(self, x):
        # input x: B, L, C
        result = x.reshape(x.size(0),
                           self.height,
                           self.width,
                           x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)

        return result

def main():
    args = get_args()

    if args.layer == None:
        raise NameError("Please ensure the parameter '--layer' is not None!\n e.g. --layer=model.net.decoderhead.LHBlock2.mlp_l")
        
    cfg = Config.fromfile(args.config)

    model = myTrain.load_from_checkpoint(cfg.test_ckpt_path, cfg = cfg)
    model = model.to('cuda')

    test_loader = build_dataloader(cfg.dataset_config, mode='test')

    if args.output_dir:
        base_dir = args.output_dir
    else:
        base_dir = os.path.dirname(cfg.test_ckpt_path)
    gradcam_output_dir = os.path.join(base_dir, "grad_cam", args.layer) 
    if os.path.exists(gradcam_output_dir):
        raise NameError("Please ensure gradcam_output_dir does not exist!")
    
    os.makedirs(gradcam_output_dir)

    for input in tqdm(test_loader):
        target_layers = [eval(args.layer)] # name of the network layer
        mask, img_id =  input[2].cuda(), input[3]

        cam = GradCAM(cfg, model=model.net, target_layers=target_layers, use_cuda=True, 
                      reshape_transform=ResizeTransform(im_h=args.imgsize, im_w=args.imgsize))
        target_category = 1  # tabby, tabby cat

        grayscale_cam_all = cam(input_tensor=(input[0], input[1]), target_category=target_category)

        for i in range(grayscale_cam_all.shape[0]):
            grayscale_cam = grayscale_cam_all[i, :]
            visualization = show_cam_on_image(0,
                                            grayscale_cam,
                                            use_rgb=True)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(visualization)
            # ax = fig.add_subplot(122)
            # ax.imshow(mask[i].cpu().numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.savefig(os.path.join(gradcam_output_dir, '{}.png'.format(img_id[i])))
            plt.close()


if __name__ == '__main__':
    main()
