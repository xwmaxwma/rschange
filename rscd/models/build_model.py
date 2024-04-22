import torch
from torch import nn
import sys
sys.path.append('rscd')
from utils.build import build_from_cfg

class myModel(nn.Module):
    def __init__(self, cfg):
        super(myModel, self).__init__()
        self.backbone = build_from_cfg(cfg.backbone)
        self.decoderhead = build_from_cfg(cfg.decoderhead)
    
    def forward(self, x1, x2, gtmask=None):
        backbone_outputs = self.backbone(x1, x2)
        if gtmask == None:
            x_list = self.decoderhead(backbone_outputs)
        else:
            x_list = self.decoderhead(backbone_outputs, gtmask)
        return x_list

"""
对于不满足该范式的模型可在backbone部分进行定义, 并在此处导入
"""

# model_config
def build_model(cfg):
    c = myModel(cfg)
    return c


if __name__ == "__main__":
    x1 = torch.randn(4, 3, 512, 512)
    x2 = torch.randn(4, 3, 512, 512)
    target = torch.randint(low=0,high=2,size=[4, 512, 512])
    file_path = r"E:\zjuse\2308CD\rschangedetection\configs\SARASNet.py"

    from utils.config import Config
    from rscd.losses import build_loss

    cfg = Config.fromfile(file_path)
    net = build_model(cfg.model_config)
    res = net(x1, x2)
    print(res.shape)
    loss = build_loss(cfg.loss_config)

    compute = loss(res,target)
    print(compute)