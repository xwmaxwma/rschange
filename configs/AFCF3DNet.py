net = 'AFCF3DNet'
######################## base_config #########################
epoch = 200
gpus = [0]
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
logging_interval = 'epoch'
resume_ckpt_path = None
pretrained_ckpt_path = None
monitor_val = 'val_change_f1'
monitor_test = ['test_change_f1']
argmax = False

test_ckpt_path = None

exp_name = 'CLCD_BS4_epoch200/{}'.format(net)

######################## dataset_config ######################
_base_ = [
    './_base_/CLCD_config.py',
]
num_class = 2
ignore_index = 255

######################### model_config #########################
model_config = dict(
    backbone = dict(
        type = 'AFCD3D_backbone'
    ),
    decoderhead = dict(
        type = 'AFCD3D_decoder',
        channel = 32
    )
)
loss_config = dict(
    type = 'myLoss',
    loss_name = ['BCEDICE_loss'],
    loss_weight = [1],
    param = dict(
        BCEDICE_loss = dict()
    )
)

######################## optimizer_config ######################
optimizer_config = dict(
    optimizer = dict(
        type = 'AdamW',
        lr = 1e-4, 
        weight_decay = 1e-4,
        lr_mode = "single"
    ),
    scheduler = dict(
        type = 'CosineAnnealingLR',
        max_epoch = epoch
    )
)

metric_cfg1 = dict(
    task = 'multiclass',
    average='micro',
    num_classes = num_class, 
)

metric_cfg2 = dict(
    task = 'multiclass',
    average='none',
    num_classes = num_class, 
)
