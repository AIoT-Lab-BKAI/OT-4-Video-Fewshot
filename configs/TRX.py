from easydict import EasyDict as edict
sampler = dict(
    n_way = 5,
    n_shot = 5,
    n_query = 5
)
dataloader = dict(
    train = dict(
        dataset = dict(
            data_path = 'splits/ucf_ARN/trainlist03.txt',
            data_dir='data/ucf101',
            img_size=224,
            seq_len=8,
            train=True,
        ),
        sampler = dict(
            **sampler,
            n_tasks = 20000,
        )
    ),
    val = dict(
        dataset = dict(
            data_path = 'splits/ucf_ARN/vallist03.txt',
            data_dir='data/ucf101',
            img_size=224,
            seq_len=8,
            train=False,
        ),
        sampler = dict(
            **sampler,
            n_tasks = 500,
        )
    ),
    test = dict(
        dataset = dict(
            data_path = 'splits/ucf_ARN/testlist03.txt',
            data_dir='data/ucf101',
            img_size=224,
            seq_len=8,
            train=False,
        ),
        sampler = dict(
            **sampler,
            n_tasks = 10000,
        )
    )
)
model = edict(
    trans_linear_in_dim=2048,
    trans_linear_out_dim = 1152,
    way = 5,
    shot = 5,
    query_per_class = 5,
    trans_dropout = 0.1,
    seq_len = 8 ,
    img_size = 224,
    method = "resnet50",
    num_gpus = 1,
    temp_set = [2,3],
    n_way = 5,
    n_shot = 5,
    n_query = 5,
)

plmodule = dict(
    lr = 0.001,
)

logger = dict(
    entity = 'aiotlab',
    project = 'few-shot-action-recognition',
    group = 'trx',
)