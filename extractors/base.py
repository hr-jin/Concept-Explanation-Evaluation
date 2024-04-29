from abc import *

class BaseExtractor(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @classmethod
    @abstractmethod
    def code(cls):
        pass
    
    @classmethod
    @abstractmethod
    def activation_func(self, **kwargs):
        pass
    
    @classmethod
    @abstractmethod
    def hidden_state_func(self, **kwargs):
        pass
    
    def load_from_file(*args,**kwargs):
        pass
    
    @classmethod
    @abstractmethod
    def extract_concepts(self):
        pass
    
    @classmethod
    @abstractmethod
    def get_concepts(self):
        pass

