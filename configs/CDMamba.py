net = 'CDMamba'
######################## base_config #########################
epoch = 200
gpus = [0]
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
logging_interval = 'epoch'
resume_ckpt_path = None
monitor_val = 'val_change_f1'
monitor_test = ['test_change_f1']
argmax = True

test_ckpt_path = None

exp_name = 'DSIFN_BS8_epoch{}/{}'.format(epoch, net)

######################## dataset_config ######################
_base_ = [
    './_base_/DSIFN_config.py',
]
num_class = 2
ignore_index = 255

######################### model_config ###########ß##############
model_config = dict(
    backbone = dict(
        type = 'CDMamba',
    init_filters = 16,
    mode = "AGLGF",
    conv_mode = "orignal_dinner",
    local_query_model = "orignal_dinner",
    up_mode = "SRCM",
    up_conv_mode = "deepwise",
    spatial_dims = 2,
    in_channels = 3,
    resdiual = False,
    blocks_down = [1, 2, 2, 4],
    blocks_up = [1, 1, 1],
    diff_abs = "later",
    stage = 2,
    mamba_act = "relu",
    norm = ["GROUP", {"num_groups": 8}]
    ),
    decoderhead = dict(
        type = 'none_class',
    )
)
loss_config = dict(
    type = 'myLoss',
    loss_name = ['CELoss', 'LOVASZ'],  
    loss_weight = [1.0, 1.0],          
    param = dict(
        CELoss = dict(ignore_index=255),
        LOVASZ = dict()
    )
)

######################## optimizer_config ######################
optimizer_config = dict(
    optimizer = dict(
        type = 'AdamW',
        lr = 1e-3, 
        weight_decay = 1e-3,
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
