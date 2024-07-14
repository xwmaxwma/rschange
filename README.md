![cap](./cap.jpg)

## 📷 Introduction

**rschange** is an open-source change detection toolbox, which is dedicated to reproducing and developing advanced methods for change detection of remote sensing images.

- Supported Methods
  - [STNet](https://ieeexplore.ieee.org/abstract/document/10219826) (ICME2023)

  - [DDLNet](https://arxiv.org/abs/2406.13606) (ICME2024 oral)

  - [CDMask](https://arxiv.org/abs/2406.15320) (Under review)
	
  - CD-Mamba (Under review, updated soon, refer to [this](https://github.com/Casey-bit))
  
  - Other popular methods including 
  
    [BIT](https://ieeexplore.ieee.org/abstract/document/9491802?casa_token=ERmQ_XMsoXcAAAAA:LS9RWgxMQwA6QT3H4lTA2uj44iYRSXkYGqFXy3c_sTujSQRGr14wOH8h7xiKqYQftXNeXG5voBVJ8g) (TGRS2021),	[SNUNet](https://ieeexplore.ieee.org/abstract/document/9355573?casa_token=NAEi6I-AywwAAAAA:uQWgA3jLiaOThIibhZneEuskdI_sDwveSliJc4pWqYiKdMcfhOQ7dzgNxJVNVL9g3vya1Vw6H39_mw) (GRSL2021),	[ChangeFormer](https://ieeexplore.ieee.org/abstract/document/9883686?casa_token=A8uGOuuiaOoAAAAA:DQUwAvWmmEaR3XY7pmwMvI2TPl5nODPAEGiDwEotvbZI_81deQwmlG619R0HEFPKHlRurTP0kWeozA) (IGARSS2022), 
  
    [LGPNet](https://ieeexplore.ieee.org/abstract/document/9627698?casa_token=i-VH46OEnuIAAAAA:wgXT7tiUYOiS-_694aYKjdeO7lQQtHyayBXUQMqCM4nWZ-iJ5rrONql4n8vpupMTsVg9jstmHK3juQ)(TGRS2021),	[SARAS-Net](https://ojs.aaai.org/index.php/AAAI/article/view/26660) (AAAI2023)
  
- Supported Datasets
  - [LEVIR-CD](https://chenhao.in/LEVIR/)
  - [DSIFN-CD](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset)
  - [WHU-CD](http://gpcv.whu.edu.cn/data/building_dataset.html)
  - [CLCD](https://github.com/liumency/CropLand-CD)
  - [SYSU-CD](https://github.com/liumency/SYSU-CD)
  
- Supported Tools
  - Training
  - Testing
  - Params and FLOPs counting
  - Class activation maps

## 🔥 News

- `2024/07/14`: Class activation maps and some other popular methods (BIT, SNUNet, ChangeFormer, LGPNet, SARAS-Net) are now supported.

- `2024/06/24`: CDMask has been submitted to Arxiv, see [here](https://arxiv.org/abs/2406.15320), and the official implementation of CDMask is available!

## 🔐 Preparation

- Environment preparation

  ```shell
  conda create -n rscd python=3.9
  conda activate rscd
  conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
  pip install -r requirements.txt
  ```

	Note: same as [rsseg](https://github.com/xwmaxwma/rssegmentation). If you have already installed the environment of [rsseg](https://github.com/xwmaxwma/rssegmentation), use it directly.

- Dataset preprocessing

  **LEVIR-CD**：The original images are sized at 1024x1024. Following its original division method, we crop these images into non-overlapping patches of 256x256.

  **WHU-CD**: It contains a pair of dual-time aerial images measuring 32507 × 15354. These images are cropped into patches of 256 × 256 size. The dataset is then randomly divided into three subsets: the training set, the validation set, and the test set, following a ratio of 8:1:1. 
  
  **DSIFN-CD & CLCD & SYSU-CD**: They all follow the original image size and dataset division method.
  
  Note: We also provide the pre-processed data, which can be downloaded at this [link](https://drive.google.com/drive/folders/1zxhJ7v3UPgNsKkdvkYCOW7DdKDAAy_ll?usp=sharing)
  
## 📒 Folder Structure

  Prepare the following folders to organize this repo:

      rschangedetection
          ├── rscd (code)
          ├── work_dirs (save the model weights and training logs)
          │   └─CLCD_BS4_epoch200 (dataset)
          │       └─stnet (model)
          │           └─version_0 (version)
          │              │  └─ckpts
          │              │      ├─test (the best ckpts in test set)
          │              │      └─val (the best ckpts in validation set)
          │              ├─log (tensorboard logs)
          │              ├─train_metrics.txt (train & val results per epoch)
          │              ├─test_metrics_max.txt (the best test results)
          │              └─test_metrics_rest.txt (other test results)
          └── data
              ├── LEVIR_CD
              │   ├── train
              │   │   ├── A
              │   │   │   └── images1.png
              │   │   ├── B
              │   │   │   └── images2.png
              │   │   └── label
              │   │       └── label.png
              │   ├── val (the same with train)
              │   └── test(the same with train)
              ├── DSIFN
              │   ├── train
              │   │   ├── t1
              │   │   │   └── images1.jpg
              │   │   ├── t2
              │   │   │   └── images2.jpg
              │   │   └── mask
              │   │       └── mask.png
              │   ├── val (the same with train)
              │   └── test
              │       ├── t1
              │       │   └── images1.jpg
              │       ├── t2
              │       │   └── images2.jpg
              │       └── mask
              │           └── mask.tif
              ├── WHU_CD
              │   ├── train
              │   │   ├── image1
              │   │   │   └── images1.png
              │   │   ├── image2
              │   │   │   └── images2.png
              │   │   └── label
              │   │       └── label.png
              │   ├── val (the same with train)
              │   └── test(the same with train)
              ├── CLCD (the same with WHU_CD)
              └── SYSU_CD
                  ├── train
                  │   ├── time1
                  │   │   └── images1.png
                  │   ├── time2
                  │   │   └── images2.png
                  │   └── label
                  │       └── label.png
                  ├── val (the same with train)
                  └── test(the same with train)

  

## 📚 Use example

- Training

  ```shell
  python train.py -c configs/STNet.py
  ```

- Testing

  ```shell
  python test.py \
  -c configs/STNet.py \
  --ckpt work_dirs/CLCD_BS4_epoch200/stnet/version_0/ckpts/test/epoch=45.ckpt \
  --output_dir work_dirs/CLCD_BS4_epoch200/stnet/version_0/ckpts/test \
  ```

- Count params and flops

  ```shell
  python tools/params_flops.py --size 256
  ```


- Class activation maps

  ```shell
  python tools/grad_cam_CNN.py -c configs/cdmask.py
  ```

  

## 🌟 Citation

If you are interested in our work, please consider giving a 🌟 and citing our work below. We will update **rschange** regularly.

```
@inproceedings{stnet,
  title={STNet: Spatial and Temporal feature fusion network for change detection in remote sensing images},
  author={Ma, Xiaowen and Yang, Jiawei and Hong, Tingfeng and Ma, Mengting and Zhao, Ziyan and Feng, Tian and Zhang, Wei},
  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={2195--2200},
  year={2023},
  organization={IEEE}
}

@article{ddlnet,
  title={DDLNet: Boosting Remote Sensing Change Detection with Dual-Domain Learning},
  author={Ma, Xiaowen and Yang, Jiawei and Che, Rui and Zhang, Huanting and Zhang, Wei},
  journal={arXiv preprint arXiv:2406.13606},
  year={2024}
}

@article{cdmask,
  title={Rethinking Remote Sensing Change Detection With A Mask View},
  author={Ma, Xiaowen and Wu, Zhenkai and Lian, Rongrong and Zhang, Wei and Song, Siyang},
  journal={arXiv preprint arXiv:2406.15320},
  year={2024}
}
```

## 💡 Acknowledgement

Thanks to previous open-sourced repo:

- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [pytorch lightning](https://lightning.ai/)
- [fvcore](https://github.com/facebookresearch/fvcore)

Thanks to the main contributor [Zhenkai Wu](https://github.com/Casey-bit)
