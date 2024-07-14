net = 'CFNet'
######################## base_config #########################
epoch = 200
gpus = [0]
save_top_k = 10
save_last = True
check_val_every_n_epoch = 1
logging_interval = 'epoch'
resume_ckpt_path = None
pretrained_ckpt_path = None
monitor_val = 'val_change_f1'
monitor_test = ['test_change_f1']
argmax = True

test_ckpt_path = None

exp_name = 'CLCD_BS4_epoch200/{}'.format(net)

######################## dataset_config ######################
_base_ = [
    './_base_/CLCD_config.py',
]
num_class = 2

######################### model_config #########################
model_config = dict(
    backbone = dict(
        type = 'ChangeFormer_EN',
        input_nc=3, 
        output_nc=2
    ),
    decoderhead = dict(
        type = 'ChangeFormer_DE',
        output_nc=2, 
        decoder_softmax=False, 
        embed_dim=256
    )
)
loss_config = dict(
    type = 'myLoss',
    loss_name = ['CELoss'],
    loss_weight = [1],
    param = dict(
        CELoss = dict(
            ignore_index=255, 
            reduction='mean'
        )
    )
)

######################## optimizer_config ######################
optimizer_config = dict(
    optimizer = dict(
        type = 'AdamW', 
        lr = 1e-4, 
        weight_decay = 1e-2,
        lr_mode = "single"
    ),
    scheduler = dict(
        type = 'linear',
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