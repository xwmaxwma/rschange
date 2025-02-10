net = 'ChangeMamba'
######################## base_config #########################
epoch = 200
gpus = [0]
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
logging_interval = 'epoch'
resume_ckpt_path = None
monitor_val = 'val_change_f1'
monitor_test = ['test_change_f1']
argmax = True

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
        type = 'CMBackbone',
        pretrained=None,
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 9, 2], 
        dims=96, 
        # ===================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate= 0.0,
        ssm_init="v0",
        forward_type="v2",
        # ===================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        # ===================
        drop_path_rate=0.1,
        patch_norm=True,
        norm_layer="ln",
        downsample_version="v2",
        patchembed_version="v2",
        gmlp=False,
        use_checkpoint=False,
    ),
    decoderhead = dict(
        type = 'CMDecoder',
        pretrained=None,
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 9, 2], 
        dims=96, 
        # ===================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate= 0.0,
        ssm_init="v0",
        forward_type="v2",
        # ===================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        # ===================
        drop_path_rate=0.1,
        patch_norm=True,
        norm_layer="ln2d",
        downsample_version="v2",
        patchembed_version="v2",
        gmlp=False,
        use_checkpoint=False,
    )
)
loss_config = dict(
    type = 'myLoss',
    loss_name = ['FocalLoss', 'LOVASZ'],
    loss_weight = [1, 0.75],
    param = dict(
        FocalLoss = dict(
            gamma=0, 
            alpha=None
        ),
        LOVASZ = dict()
    )
)

######################## optimizer_config ######################
optimizer_config = dict(
    optimizer = dict(
        type = 'AdamW',
        lr = 1e-4, 
        weight_decay = 5e-4,
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
