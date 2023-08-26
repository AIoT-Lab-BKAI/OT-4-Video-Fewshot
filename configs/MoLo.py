from easydict import EasyDict as edict
from pytorch_lightning.strategies import DeepSpeedStrategy

sampler_common = dict(
    n_way = 5,
    n_shot = 1,
    n_query = 6,
)
ds_common = dict(
    root_dir='data/ucf101',
    img_size=224,
    seq_len=8,
    class_id_path='splits/classes.json',
)
sampler = dict(
    train = dict(
        **sampler_common,
        n_tasks=15000,
    ),
    val = dict(
        **sampler_common,
        n_tasks=500,
    ),
    test = dict(
        **sampler_common,
        n_tasks=10000,
    ),
)
dataset = dict(
    train = dict(
        split_path = 'splits/ucf_ARN/trainlist03.txt',
        **ds_common,
        train=True,
    ),
    val = dict(
        split_path = 'splits/ucf_ARN/vallist03.txt',
        **ds_common,
    ),
    test = dict(
        split_path = 'splits/ucf_ARN/testlist03.txt',
        **ds_common,
    ),
)
model = dict(
    USE_CONTRASTIVE = True,
    USE_CLASSIFICATION_VALUE= 0.8,
    USE_RECONS= True,

    DATA = dict(
        NUM_INPUT_FRAMES = 8
    ),
    USE_OT = False
)

pl_module = dict(
    recons_coef = 0.1,
    contrastive_coef = 0.05,
    optimizer=dict(
        weight_decay = 5e-4,
    ),
    bn = dict(
        weight_decay = 0,
    ),
    accum_grad = 8,
    log_freq = 50,
)

logger = dict(
    entity = 'aiotlab',
    project = 'few-shot-action-recognition',
    group = 'molo',
)
checkpoint = dict(
    save_top_k=1,
    monitor='val_acc',
    mode='max',
    save_last=True
)
trainer = dict(
    devices=1,
    accelerator='gpu',
    strategy = DeepSpeedStrategy(stage=3, offload_optimizer=True,offload_parameters=True),
    
    max_epochs=1,
    
    # check_val_every_n_epoch=None,
    val_check_interval=500,
    
    log_every_n_steps=50,
)
optimizer = dict(
)
