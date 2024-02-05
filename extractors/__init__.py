from .ae import AutoEncoder
from .tcav import TCAVExtractor
from .neuron import Neuron

EXTRACTORS = {
    AutoEncoder.code(): AutoEncoder,
    TCAVExtractor.code(): TCAVExtractor,
    Neuron.code(): Neuron,
}

def extractor_factory(cfg, dataloader):
    extractor = EXTRACTORS[cfg['extractor']]
    if cfg['load_extractor']:
        return extractor
    return extractor(cfg, dataloader).to(cfg['device'])