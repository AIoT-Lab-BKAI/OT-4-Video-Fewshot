import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from utils.utils import get_wandb_logger

ds_common = {
    'root_dir': 'data/ucf101',
    'class_id_path': 'splits/classes.json',
}
dataset = dict(
    train = dict(
        **ds_common,
        train=True,
        split_path='splits/classification/train.txt'
    ),
    val = dict(
        **ds_common,
        split_path='splits/classification/val.txt'
    ),
    test = dict(
        **ds_common,
        split_path='splits/classification/test.txt'
    ),
)
logger_cfg = dict(
    entity = 'aiotlab',
    project = 'c3d_classification',
    group = 'c3d',
)

logger = True
logger = get_wandb_logger(logger_cfg) if logger else None

cbs = []
ckpt_cb = pl.callbacks.ModelCheckpoint(
x
    dirpath='workdirs/c3d'
)
cbs.append(ckpt_cb)

lr_monitor = LearningRateMonitor(logging_interval='step')
cbs.append(lr_monitor)

trainer = dict(
    devices=1,
    accelerator='gpu',
    max_epochs=100,
    check_val_every_n_epoch=4,
    callbacks=cbs,
    logger=logger,
)
