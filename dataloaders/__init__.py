from .tcav import TCAVDataloader
from .ae import AEDataloader
from .conceptx import ConceptXDataloader
from .conceptx_naive import ConceptXNaiveDataloader


DATALOADERS = {
    "spine": AEDataloader,
    AEDataloader.code(): AEDataloader,
    ConceptXDataloader.code(): ConceptXDataloader,
    TCAVDataloader.code(): TCAVDataloader,
    ConceptXNaiveDataloader.code(): ConceptXNaiveDataloader,
}


def dataloader_factory(cfg, data, model):

    dataloader = DATALOADERS[cfg["extractor"]]
    train_dataloader = dataloader(cfg, data, model)
    return train_dataloader
