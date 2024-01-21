from abc import *


class BaseModel(metaclass=ABCMeta):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @classmethod
    @abstractmethod
    def code(cls):
        pass

