from abc import *

class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, cfg):
        self.cfg = cfg

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def load_dataset(self):
        pass