# optimizer
optimizer = dict(type='AdamW', lr=4e-4, weight_decay=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.01,
    min_lr_ratio=1e-5,
    periods=[5868, 5868, 7824],
    restart_weights=[0.8, 0.6, 0.5],
    by_epoch=False
    )
runner = dict(type='EpochBasedRunner', max_epochs=20)