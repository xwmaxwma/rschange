import torch
import torch.optim as optim
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import math

class lambdax:
    def __init__(self, cfg):
        self.cfg = cfg
    @staticmethod
    def lambda_epoch(self, epoch):
        return math.pow(1 - epoch / self.cfg.max_epoch, self.cfg.poly_exp)


def get_optimizer(cfg, net):
    if cfg.lr_mode == 'multi':
        layerwise_params = {"backbone.*": dict(lr=cfg.backbone_lr, weight_decay=cfg.backbone_weight_decay)}
        net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
    else:
        net_params = net.parameters()

    if cfg.type == "AdamW":
        optimizer = optim.AdamW(net_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        # optimizer = Lookahead(optimizer)
    elif cfg.type == "SGD":
        optimizer = optim.SGD(net_params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum,
                              nesterov=False)
    else:
        raise KeyError("The optimizer type ( %s ) doesn't exist!!!" % cfg.type)

    return optimizer

def get_scheduler(cfg, optimizer):
    if cfg.type == 'Poly':
        lambda1 = lambda epoch: math.pow(1 - epoch / cfg.max_epoch, cfg.poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif cfg.type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epoch, eta_min=1e-6)
    elif cfg.type == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(cfg.max_epoch + 1)
            return lr_l
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif cfg.type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    elif cfg.type == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)
    elif cfg.type == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=cfg.patience, factor=cfg.factor)
    else:
        raise KeyError("The scheduler type ( %s ) doesn't exist!!!" % cfg.type)
    
    return scheduler

def build_optimizer(cfg, net):
    optimizer = get_optimizer(cfg.optimizer, net)
    scheduler = get_scheduler(cfg.scheduler, optimizer)
    # if cfg.type == 'Poly':
    #     lambda1 = lambda epoch: math.pow(1 - epoch / cfg.max_epoch, cfg.poly_exp)
    #     scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    # elif cfg.type == 'CosineAnnealingLR':
    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epoch, eta_min=1e-6)
    # else:
    #     raise KeyError("The scheduler type ( %s ) doesn't exist!!!" % cfg.type)
    
    return optimizer, scheduler


    

    
    
