![cap](./cap.jpg)

## 📷 Introduction

**rschange** is an open-source change detection toolbox, which is dedicated to reproducing and developing advanced methods for change detection of remote sensing images.

- Supported Methods
  - [STNet](https://ieeexplore.ieee.org/abstract/document/10219826) (ICME2023)

  - [DDLNet](https://arxiv.org/abs/2406.13606) (ICME2024 oral)

  - [CDMask](https://arxiv.org/abs/2406.15320) (Under review)
 
  - [STeinFormer](https://ieeexplore.ieee.org/document/10815617) (JSTARS2025)
	
  - [CD-Lamba](https://arxiv.org/abs/2501.15455) (Under review)
 
  - [CDXLSTM](https://arxiv.org/abs/2411.07863) (GRSL2025)
  
  - Other popular methods including 
  
    [BIT](https://ieeexplore.ieee.org/abstract/document/9491802?casa_token=ERmQ_XMsoXcAAAAA:LS9RWgxMQwA6QT3H4lTA2uj44iYRSXkYGqFXy3c_sTujSQRGr14wOH8h7xiKqYQftXNeXG5voBVJ8g) (TGRS2021),	[SNUNet](https://ieeexplore.ieee.org/abstract/document/9355573?casa_token=NAEi6I-AywwAAAAA:uQWgA3jLiaOThIibhZneEuskdI_sDwveSliJc4pWqYiKdMcfhOQ7dzgNxJVNVL9g3vya1Vw6H39_mw) (GRSL2021),	[ChangeFormer](https://ieeexplore.ieee.org/abstract/document/9883686?casa_token=A8uGOuuiaOoAAAAA:DQUwAvWmmEaR3XY7pmwMvI2TPl5nODPAEGiDwEotvbZI_81deQwmlG619R0HEFPKHlRurTP0kWeozA) (IGARSS2022), 
  
    [LGPNet](https://ieeexplore.ieee.org/abstract/document/9627698?casa_token=i-VH46OEnuIAAAAA:wgXT7tiUYOiS-_694aYKjdeO7lQQtHyayBXUQMqCM4nWZ-iJ5rrONql4n8vpupMTsVg9jstmHK3juQ)(TGRS2021),	[SARAS-Net](https://ojs.aaai.org/index.php/AAAI/article/view/26660) (AAAI2023),   [USSFCNet](https://ieeexplore.ieee.org/document/10081023) (TGRS2023), [AFCF3DNet](https://ieeexplore.ieee.org/abstract/document/10221754) (TGRS2023), [RSMamba](https://arxiv.org/abs/2403.19654) (GRSL2024), [ChangeMamba](https://ieeexplore.ieee.org/document/10565926/) (TGRS2024)
  
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
- `2025/04/16`: [CDXLSTM](https://arxiv.org/abs/2411.07863) has been accepted by GRSL2025.

- `2025/03/13`: The official files of the environment preparation are now available in [rscd_mamba](https://drive.google.com/drive/folders/1p0bGAzQX6HkcbTRS5q-ynuhTLQukDLaH?usp=sharing).
   
- `2025/02/11`: The official implementation of [CD-Lamba](https://arxiv.org/abs/2501.15455), [CDXLSTM](https://arxiv.org/abs/2411.07863) and some other popular methods (RSMamba, ChangeMamba) are now available.

- `2025/01/02`: [STeInFormer](https://ieeexplore.ieee.org/document/10815617) has been accepted by JSTARS2025.

- `2024/07/14`: Class activation maps and some other popular methods (BIT, SNUNet, ChangeFormer, LGPNet, SARAS-Net) are now supported.

- `2024/06/24`: CDMask has been submitted to Arxiv, see [here](https://arxiv.org/abs/2406.15320), and the official implementation of CDMask is available!

## 🔐 Preparation

- Environment preparation

  ```shell
	conda create --name rscd python=3.8
	conda activate rscd
	conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
	pip install pytorch-lightning==2.0.5
	pip install scikit-image==0.19.3 numpy==1.24.4
	pip install torchmetrics==1.0.1
	pip install -U catalyst==20.09
	pip install albumentations==1.3.1
	pip install einops==0.6.1
	pip install timm==0.6.7
	pip install addict==2.4.0
	pip install soundfile==0.12.1
	pip install ttach==0.0.3
	pip install prettytable==3.8.0
	pip install -U openmim
	pip install triton==2.0.0
	mim install mmcv
	pip install -U fvcore
  ```
	If you need to run a model based on Mamba, please additionally download the [releases](https://github.com/xwmaxwma/rschange/releases/tag/mamba_env) and then perform the following installation for the Mamba environment.
  ```shell
  pip install causal_conv1d-1.2.0.post1+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
  pip install mamba_ssm-1.2.0.post1+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
  ```

	[Optional] We have also prepared compressed files [rscd_mamba](https://drive.google.com/drive/folders/1p0bGAzQX6HkcbTRS5q-ynuhTLQukDLaH?usp=sharing) for the CD-Lamba's environment, which you can download directly and install according to the following instructions.

  ```shell
  // Firstly, you must be in a Linux environment (Ubuntu in Linux or WSL2 in windows).
  // Then, place this compressed file in the folder of \home\xxx\anaconda3\envs\
  // Finally,
  mkdir -p rscd_mamba && tar -xzf rscd_mamba.tar.gz -C rscd_mamba
  conda activate rscd_mamba
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
  python tools/grad_cam_CNN.py -c configs/cdxformer.py --layer=model.net.decoderhead.LHBlock2.mlp_l
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

@INPROCEEDINGS{ddlnet,
  author={Ma, Xiaowen and Yang, Jiawei and Che, Rui and Zhang, Huanting and Zhang, Wei},
  booktitle={2024 IEEE International Conference on Multimedia and Expo (ICME)}, 
  title={DDLNet: Boosting Remote Sensing Change Detection with Dual-Domain Learning}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/ICME57554.2024.10688140}}

@article{cdmask,
  title={Rethinking Remote Sensing Change Detection With A Mask View},
  author={Ma, Xiaowen and Wu, Zhenkai and Lian, Rongrong and Zhang, Wei and Song, Siyang},
  journal={arXiv preprint arXiv:2406.15320},
  year={2024}
}

@ARTICLE{steinformer,
  author={Ma, Xiaowen and Wu, Zhenkai and Ma, Mengting and Zhao, Mengjiao and Yang, Fan and Du, Zhenhong and Zhang, Wei},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={STeInFormer: Spatial–Temporal Interaction Transformer Architecture for Remote Sensing Change Detection}, 
  year={2025},
  volume={18},
  number={},
  pages={3735-3745},
  doi={10.1109/JSTARS.2024.3522329}}
```

## 📮 Contact

If you are confused about the content of our paper or look forward to further academic exchanges and cooperation, please do not hesitate to contact us. The e-mail address is xwma@zju.edu.cn. We look forward to hearing from you!

## 💡 Acknowledgement

Thanks to previous open-sourced repo:

- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [pytorch lightning](https://lightning.ai/)
- [fvcore](https://github.com/facebookresearch/fvcore)

Thanks to the main contributor [Zhenkai Wu](https://github.com/Casey-bit)
