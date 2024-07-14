net = 'maskcd'
######################## base_config #########################
epoch = 200
gpus = [0]
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
logging_interval = 'epoch'
resume_ckpt_path = None
monitor_val = 'val_change_f1'
monitor_test = ['test_change_f1_0',
                'test_change_f1_1',
                'test_change_f1_2',
                'test_change_f1_3',
                'test_change_f1_4',
                'test_change_f1_5',
                'test_change_f1_6']
argmax = False

test_ckpt_path = r'work_dirs\CLCD_BS4_epoch200\maskcd\version_0\ckpts\test\test_change_f1_0\test_change_f1_4=0.7855-epoch=117.ckpt'

exp_name = 'CLCD_BS4_epoch{}/{}'.format(epoch, net)

######################## dataset_config ######################
_base_ = [
    './_base_/CLCD_config.py',
]
num_class = 2
ignore_index = 255

######################### model_config #########################
model_config = dict(
    backbone = dict(
        type = 'Base',
        name = 'Seaformer'
    ),
    decoderhead = dict(
        type = 'CDMask',
        channels = [64, 128, 192, 256],
        num_classes = num_class - 1,
        num_queries = 5,
        dec_layers = 14
    )
)
loss_config = dict(
    type = 'myLoss',
    loss_name = ['Mask2formerLoss'],
    loss_weight = [1],
    param = dict(
        Mask2formerLoss = dict(
            class_weight=1.0,
            dice_weight=10.0,
            mask_weight=10.0,
            no_object_weight=0.1,
            dec_layers = 14,
            num_classes=num_class - 1,
            device="cuda:{}".format(gpus[0])
        )
    )
)

######################## optimizer_config ######################
optimizer_config = dict(
    optimizer = dict(
        type = 'AdamW',
        backbone_lr = 0.0005,
        backbone_weight_decay = 0.01,
        lr = 0.0001,
        weight_decay = 0.05,
        lr_mode = "multi"
    ),
    scheduler = dict(
        type = 'Poly',
        poly_exp = 0.9,
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