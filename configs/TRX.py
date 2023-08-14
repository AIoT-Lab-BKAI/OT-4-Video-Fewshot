from easydict import EasyDict as edict
from pytorch_lightning.strategies import DeepSpeedStrategy

sampler_common = dict(
    n_way = 5,
    n_shot = 5,
    n_query = 5,
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
        n_tasks=20000,
    ),
    val = dict(
        **sampler_common,
        n_tasks=500,
    ),
    test = dict(
        **sampler_common,
        n_tasks=1000,
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
    trans_linear_in_dim=2048,
    trans_linear_out_dim = 1152,
    
    way = sampler_common['n_way'],
    shot = sampler_common['n_shot'],
    query_per_class = sampler_common['n_query'],
    
    trans_dropout = 0.1,
    seq_len = ds_common['seq_len'],
    img_size = ds_common['img_size'],
    
    method = "resnet50",
    temp_set = [2,3],
    
    n_way = sampler_common['n_way'],
    n_shot = sampler_common['n_shot'],
    n_query = sampler_common['n_query'],
)

plmodule = dict(
    lr = 0.001,
)

logger = dict(
    entity = 'aiotlab',
    project = 'few-shot-action-recognition',
    group = 'trx',
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
    
    check_val_every_n_epoch=None,
    val_check_interval=1000,

    accumulate_grad_batches=4,
    log_every_n_steps=10,
    precision=16,
)
    
