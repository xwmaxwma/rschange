from PIL import Image
from .base_dataset import BaseDataset
import numpy as np
class CLCD(BaseDataset):
    def __init__(self, data_root='data/CLCD', mode='train', transform=None, imgA_dir='image1', imgB_dir='image2', mask_dir='label', img_suffix='.png', mask_suffix='.png', **kwargs):
        super(CLCD, self).__init__(transform, mode)
        self.imgA_dir = imgA_dir
        self.imgB_dir = imgB_dir
        self.img_suffix = img_suffix
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix

        self.data_root = data_root + "/" + mode
        self.file_paths = self.get_path(self.data_root, imgA_dir, imgB_dir, mask_dir)

        #RGB
        self.color_map = {
            'NotChanged' : np.array([0, 0, 0]),  # label 0
            'Changed' : np.array([255, 255, 255]),  # label 1
        }

        self.num_classes = 2

    def rgb2label(self,mask_rgb):
        
        mask_rgb = np.array(mask_rgb)
        _mask_rgb = mask_rgb.transpose(2, 0, 1)
        label_seg = np.zeros(_mask_rgb.shape[1:], dtype=np.uint8)
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['NotChanged'], axis=-1)] = 0
        label_seg[np.all(_mask_rgb.transpose([1, 2, 0]) == self.color_map['Changed'], axis=-1)] = 1

        _label_seg = Image.fromarray(label_seg).convert('L')
        return _label_seg


