import sys
sys.path.append('rscd')
from typing import Iterable, Optional, Sequence, Union
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t
from rscd.datasets.levircd_dataset import *
from rscd.datasets.whucd_dataset import *
from rscd.datasets.dsifn_dataset import *
from rscd.datasets.clcd_dataset import *
from rscd.datasets.sysucd_dataset import *
from rscd.datasets.base_dataset import *

def get_loader(dataset, cfg):
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle=cfg.shuffle,
        drop_last=cfg.drop_last
    )
    return loader

# dataset_config
def build_dataloader(cfg, mode='train'): # get dataloader
    dataset_type = cfg.type
    data_root = cfg.data_root
    if mode == 'train':
        dataset = eval(dataset_type)(data_root, mode, **cfg.train_mode)
        loader_cfg = cfg.train_mode.loader
    elif mode == 'val':
        dataset = eval(dataset_type)(data_root, mode, **cfg.val_mode)
        loader_cfg = cfg.val_mode.loader
    else:
        dataset = eval(dataset_type)(data_root, mode, **cfg.test_mode)
        loader_cfg = cfg.test_mode.loader

    data_loader = DataLoader(
        dataset = dataset,
        batch_size = loader_cfg.batch_size,
        num_workers = loader_cfg.num_workers,
        pin_memory = loader_cfg.pin_memory,
        shuffle = loader_cfg.shuffle,
        drop_last = loader_cfg.drop_last
    )
    
    return data_loader

if __name__ == '__main__': #you can test dataloader from here
    file_path = "E:/zjuse/2308CD/rschangedetection/configs/BIT.py"

    print(file_path)

    from utils.config import Config

    cfg = Config.fromfile(file_path)
    print(cfg)
    train_loader = build_dataloader(cfg.dataset_config)
    cnt = 0
    for i,(imgA, imgB, tar) in enumerate(train_loader):
        print(imgA.shape)
        cnt += 1
        if cnt > 10:
            break

    # print("start print val_loader #####################")
    #
    # for i,(img,tar) in enumerate(val_loader):
    #     print(img.shape)
    #     cnt += 1
    #     if cnt > 20:
    #         break
