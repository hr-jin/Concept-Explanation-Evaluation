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
    
    @staticmethod
    def get_token_freq(eval_tokens):
        token_freq_dict = {}
        for i in np.unique(eval_tokens):
            token_freq_dict[i] = (eval_tokens==i).sum().item()
        freq_threshold = np.mean(list(token_freq_dict.values()))
        return token_freq_dict, freq_threshold
    