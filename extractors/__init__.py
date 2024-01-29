from .ae import AutoEncoder
from .tcav import TCAVExtractor
from .spine import SpineExtractor
from .convex_optim import ConvexOptimExtractor

EXTRACTORS = {
    AutoEncoder.code(): AutoEncoder,
    TCAVExtractor.code(): TCAVExtractor,
    SpineExtractor.code(): SpineExtractor,
    ConvexOptimExtractor.code(): ConvexOptimExtractor,
}

def extractor_factory(cfg, dataloader):
    extractor = EXTRACTORS[cfg['extractor']]
    if cfg['load_extractor']:
        return extractor
    return extractor(cfg, dataloader).to(cfg['device'])