from .ae import AutoEncoder
from .tcav import TCAVExtractor

EXTRACTORS = {
    AutoEncoder.code(): AutoEncoder,
    TCAVExtractor.code(): TCAVExtractor,
}

def extractor_factory(cfg, dataloader):
    extractor = EXTRACTORS[cfg['extractor']]
    if cfg['load_extractor']:
        return extractor
    return extractor(cfg, dataloader).to(cfg['device'])