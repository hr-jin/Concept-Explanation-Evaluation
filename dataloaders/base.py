from abc import *

class AbstractDataloader(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def __len__(self, **kwargs):
        pass
    
    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def next(self):
        pass