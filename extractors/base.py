from abc import *

class BaseExtractor(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @classmethod
    @abstractmethod
    def code(cls):
        pass
    
    @abstractclassmethod
    def activation_func(self, **kwargs):
        pass
    
    def load_from_file():
        pass
    
    @abstractclassmethod
    def extract_concepts(self):
        pass
    
    @abstractclassmethod
    def get_concepts(self):
        pass

