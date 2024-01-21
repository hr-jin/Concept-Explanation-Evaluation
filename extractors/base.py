from abc import *

class BaseExtractor(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @classmethod
    @abstractmethod
    def code(cls):
        pass
    
    @abstractclassmethod
    def get_activations():
        pass

