from .tcav import TCAVDataloader
from .ae import AEDataloader
from .conceptx import ConceptXDataloader



DATALOADERS = {
    "neuron": AEDataloader,
    AEDataloader.code(): AEDataloader,
    ConceptXDataloader.code(): ConceptXDataloader,
    TCAVDataloader.code(): TCAVDataloader,
}


def dataloader_factory(cfg, data, model):

    dataloader = DATALOADERS[cfg["dataloader"]]
    train_dataloader = dataloader(cfg, data, model)
    return train_dataloader
