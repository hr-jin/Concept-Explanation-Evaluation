from .tcav import TCAVDataloader
from .ae import AEDataloader
from .conceptx import ConceptXDataloader
from .conceptx_naive import ConceptXNaiveDataloader


DATALOADERS = {
    "neuron": AEDataloader,
    AEDataloader.code(): AEDataloader,
    ConceptXDataloader.code(): ConceptXDataloader,
    TCAVDataloader.code(): TCAVDataloader,
    ConceptXNaiveDataloader.code(): ConceptXNaiveDataloader,
}


def dataloader_factory(cfg, data, model):

    dataloader = DATALOADERS[cfg["dataloader"]]
    train_dataloader = dataloader(cfg, data, model)
    return train_dataloader
