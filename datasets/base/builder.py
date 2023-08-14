from utils.registry import Registry
from .base_dataset import Base_Fewshot_Dataset

DATASET_REGISTRY = Registry("DATASET")

def build_dataset(dataset_name, cfg):
    """
    Builds a dataset according to the "dataset_name".
    Args:
        dataset_name (str):     the name of the dataset to be constructed.
        cfg          (Config):  global config object. 
        split        (str):     the split of the data loader.
    Returns:
        Dataset      (Dataset):    a dataset object constructed for the specified dataset_name.
    """
    name = dataset_name.capitalize()
    return DATASET_REGISTRY.get(name)(**cfg)

def build_fewshot_dataset(dataset_name, cfg, sampler_cfg):
    dataset = build_dataset(dataset_name, cfg)
    return Base_Fewshot_Dataset(dataset, **sampler_cfg)