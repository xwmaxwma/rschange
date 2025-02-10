from torch.utils.data import Dataset
from .transform import *
import albumentations as albu
from PIL import Image
import numpy as np
import os
import torch
import albumentations as A

class BaseDataset(Dataset):
    def __init__(self, transform=None,mode="train"):
        self.mosaic_ratio = 0.25
        self.mode = mode
        self.img_size = (1024,1024)
        aug_list = []
        for k,v in transform.items():
            if v != None:
                aug_list.append(eval(k)(**v))
            else: aug_list.append(eval(k)())

        self.transform = Compose(aug_list)

        self.t1_normalize = A.Compose([
            A.Normalize()
        ])

        self.t2_normalize = A.Compose([
            A.Normalize()
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        imgA, imgB, mask, img_id = self.load_img_and_mask(index)
        if len(self.transform.transforms) != 0:
            imgA, imgB, mask = self.transform([imgA, imgB], mask)
        imgA, imgB, mask = np.array(imgA), np.array(imgB), np.array(mask)
        imgA = self.t1_normalize(image=imgA)['image']
        imgB = self.t2_normalize(image=imgB)['image']
        imgA, imgB = [torch.from_numpy(img).permute(2, 0, 1).float() for img in [imgA, imgB]]
        mask = torch.from_numpy(mask).long()
        return imgA, imgB, mask, img_id

    def get_path(self, data_root, imgA_dir, imgB_dir, mask_dir):
        imgA_filename_list = os.listdir(os.path.join(data_root, imgA_dir))
        imgB_filename_list = os.listdir(os.path.join(data_root, imgB_dir))
        mask_filename_list = os.listdir(os.path.join(data_root, mask_dir))
        assert len(imgA_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids
    
    def load_img_and_mask(self, index):
        img_id = self.file_paths[index]
        imgA_name = os.path.join(self.data_root, self.imgA_dir, img_id + self.img_suffix)
        imgB_name = os.path.join(self.data_root, self.imgB_dir, img_id + self.img_suffix)
        mask_name = os.path.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        imgA = Image.open(imgA_name).convert('RGB')
        imgB = Image.open(imgB_name).convert('RGB')
        mask_rgb = Image.open(mask_name).convert('RGB')
        mask = self.rgb2label(mask_rgb)
        return imgA, imgB, mask, img_id

    # def load_mosaic_img_and_mask(self, index):
    #     indexes = [index] + [random.randint(0, len(self.file_paths) - 1) for _ in range(3)]
    #     img_a, mask_a = self.load_img_and_mask(indexes[0])
    #     img_b, mask_b = self.load_img_and_mask(indexes[1])
    #     img_c, mask_c = self.load_img_and_mask(indexes[2])
    #     img_d, mask_d = self.load_img_and_mask(indexes[3])

    #     img_a, mask_a = np.array(img_a), np.array(mask_a)
    #     img_b, mask_b = np.array(img_b), np.array(mask_b)
    #     img_c, mask_c = np.array(img_c), np.array(mask_c)
    #     img_d, mask_d = np.array(img_d), np.array(mask_d)

    #     h = self.img_size[0]
    #     w = self.img_size[1]

    #     start_x = w // 4
    #     strat_y = h // 4
    #     # The coordinates of the splice center
    #     offset_x = random.randint(start_x, (w - start_x))
    #     offset_y = random.randint(strat_y, (h - strat_y))

    #     crop_size_a = (offset_x, offset_y)
    #     crop_size_b = (w - offset_x, offset_y)
    #     crop_size_c = (offset_x, h - offset_y)
    #     crop_size_d = (w - offset_x, h - offset_y)

    #     random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
    #     random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
    #     random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
    #     random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

    #     croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
    #     croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
    #     croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
    #     croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

    #     img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
    #     img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
    #     img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
    #     img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

    #     top = np.concatenate((img_crop_a, img_crop_b), axis=1)
    #     bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
    #     img = np.concatenate((top, bottom), axis=0)

    #     top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
    #     bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
    #     mask = np.concatenate((top_mask, bottom_mask), axis=0)
    #     mask = np.ascontiguousarray(mask)
    #     img = np.ascontiguousarray(img)
    #     img = Image.fromarray(img)
    #     mask = Image.fromarray(mask)
    #     # print(img.shape)

    #     return img, mask

