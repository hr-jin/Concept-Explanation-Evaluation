from .base import BaseEvaluator
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pprint
from logger import logger
import os
import time

class Evaluator1(nn.Module, BaseEvaluator):
    """
    An unsupervised learning method that hopes to learn sparsely activated concepts through AutoEncoder
    """
    def __init__(self, cfg, activation_func, model):
        nn.Module.__init__(self)
        BaseEvaluator.__init__(self, cfg, activation_func, model)
        
    @classmethod
    def code(cls):
        return 'eva1'