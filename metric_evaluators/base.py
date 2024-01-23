from abc import *
import torch
import time
import numpy as np
from transformer_lens import utils
import pandas as pd
from tqdm import tqdm
from logger import logger
from functools import partial
from sklearn.cluster import KMeans
from sklearn import metrics
import torch.nn.functional as F
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
    