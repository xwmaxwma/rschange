net = 'RSMamba'
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
argmax = False

test_ckpt_path = None

exp_name = 'WHUCD_BS8_epoch{}/{}'.format(epoch, net)

######################## dataset_config ######################
_base_ = [
    './_base_/WHUCD_config.py',
]
num_class = 2
ignore_index = 255

######################### model_config #########################
model_config = dict(
    backbone = dict(
        type = 'RSM_CD',
        dims=96, 
        depths=[ 2, 2, 9, 2 ], 
        ssm_d_state=16, 
        ssm_dt_rank="auto", 
        ssm_ratio=2.0, 
        mlp_ratio=4.0
    ),
    decoderhead = dict(
        type = 'none_class',
    )
)
loss_config = dict(
    type = 'myLoss',
    loss_name = ['FCCDN_loss_without_seg'],
    loss_weight = [1],
    param = dict(
        FCCDN_loss_without_seg = dict()
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
        type = 'reduce',
        patience = 12,
        factor = 0.1    
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
