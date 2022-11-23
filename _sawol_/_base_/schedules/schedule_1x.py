# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.01,
    min_lr_ratio=1e-05,
    by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=12)
