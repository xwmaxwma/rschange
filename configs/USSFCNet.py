net = 'USSFCNet'
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
argmax = True

test_ckpt_path = 'ckpts/CLCD_USSFCNet_epoch200/USSFCNet_nopad/val_change_f1=0.6540-epoch=60.ckpt'

exp_name = 'LEVIRCD_BS8_epoch200/{}'.format(net)

######################## dataset_config ######################
_base_ = [
    './_base_/LEVIRCD_config.py',
]
num_class = 2
ignore_index = 255

######################### model_config #########################
model_config = dict(
    backbone = dict(
        type = 'USSFCNet_encoder',
        in_ch = 3,
        ratio = 0.5
    ),
    decoderhead = dict(
        type = 'USSFCNet_decoder',
        out_ch = 2,
        ratio = 0.5
    )
)
loss_config = dict(
    type = 'myLoss',
    loss_name = ['FocalLoss', 'dice_loss'],
    loss_weight = [0.5, 0.5],
    param = dict(
        FocalLoss = dict(
            gamma=0, 
            alpha=None
        ),
        dice_loss = dict(
            eps=1e-7
        )
    )
)

######################## optimizer_config ######################
optimizer_config = dict(
    optimizer = dict(
        type = 'SGD',
        lr = 0.05,
        momentum=0.9,
        weight_decay = 5e-5,
        lr_mode = "single"
    ),
    scheduler = dict(
        type = 'step',
        step_size = 50,
        gamma = 0.1
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