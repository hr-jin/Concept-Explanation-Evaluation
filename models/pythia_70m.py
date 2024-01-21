from .base import BaseModel
from transformer_lens import HookedTransformer

class Pythia70M(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.load_model()
    
    @classmethod
    def code(cls):
        return 'pythia-70m'
    
    def load_model(self):
        model_name = self.cfg['model_to_interpret']
        model_path = self.cfg['model_dir']
        device = self.cfg['device']
        if model_path != '':
            model = HookedTransformer.from_pretrained(model_path).to(device)
        else:
            model = HookedTransformer.from_pretrained(model_name).to(device)
        self.model = model