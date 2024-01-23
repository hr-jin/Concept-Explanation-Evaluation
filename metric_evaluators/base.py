from abc import *
from utils import *

class BaseMetricEvaluator(metaclass=ABCMeta):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @classmethod
    @abstractmethod
    def code(cls):
        pass
    
    @abstractmethod
    def get_metric():
        pass
    