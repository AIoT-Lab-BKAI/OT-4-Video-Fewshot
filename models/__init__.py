
from utils.utils import Registry
from models.C3D import C3D
from models.TRX import CNN_TRX
from models.M1 import M1


MODEL_REGISTRY = Registry()
MODEL_REGISTRY.register("C3D", C3D)
MODEL_REGISTRY.register("TRX", CNN_TRX)
MODEL_REGISTRY.register("M1", M1)

