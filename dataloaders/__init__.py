from .tcav import TCAVDataloader
from .ae import AEDataloader
from .convex_optim import ConvexOptimDataloader


DATALOADERS = {
    "spine": AEDataloader,
    "conceptx":AEDataloader,
    AEDataloader.code(): AEDataloader,
    TCAVDataloader.code(): TCAVDataloader,
    ConvexOptimDataloader.code(): ConvexOptimDataloader,
}


def dataloader_factory(cfg, data, model):
    
    dataloader = DATALOADERS[cfg['extractor']]
    train_dataloader = dataloader(cfg, data, model)
    return train_dataloader
