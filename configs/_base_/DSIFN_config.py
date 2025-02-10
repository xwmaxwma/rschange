dataset_config = dict(
    type = 'DSIFN',
    data_root = 'data/DSIFN',
    train_mode = dict(
        transform = dict(
            # RandomSizeAndCrop = {"size": 512, "crop_nopad": False},
            RandomHorizontallyFlip = {"p": 0.5},
            RandomVerticalFlip = {"p": 0.5},
            RandomGaussianBlur = None,
        ),
        loader = dict(
            batch_size = 4,
            num_workers = 4,
            pin_memory=True,
            shuffle = True,
            drop_last = True
        ),
    ),
    
    val_mode = dict(
        transform = dict(
            # RandomSizeAndCrop = {"size": 512, "crop_nopad": False},
        ),
        loader = dict(
            batch_size = 4,
            num_workers = 4,
            pin_memory=True,
            shuffle = False,
            drop_last = False
        )
    ),

    test_mode = dict(
        transform = dict(
            # RandomSizeAndCrop = {"size": 512, "crop_nopad": False},
        ),
        loader = dict(
            batch_size = 4,
            num_workers = 4,
            pin_memory=True,
            shuffle = False,
            drop_last = False
        )
    ),
)
