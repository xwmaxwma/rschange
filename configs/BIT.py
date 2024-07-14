net = 'BIT'
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
        type = 'BIT_Backbone',
        input_nc=3, 
        output_nc=32, 
        resnet_stages_num=4,
        backbone='resnet18',
        if_upsample_2x=True,
    ),
    decoderhead = dict(
        type = 'BASE_Transformer',
        input_nc=32, 
        output_nc=2, 
        token_len=4, 
        with_pos='learned', 
        enc_depth=1, 
        dec_depth=8, 
        dim_head=8,
        decoder_dim_head=8,
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