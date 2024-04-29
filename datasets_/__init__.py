from .pile import PileDataset
from .harmful_qa import HarmFulQADataset
from .conceptx_data import ConceptXData
from .conceptx_naive import ConceptXNaiveDataset

DATASETS = {
    PileDataset.code(): PileDataset,
    HarmFulQADataset.code(): HarmFulQADataset,
    ConceptXData.code(): ConceptXData,
    ConceptXNaiveDataset.code(): ConceptXNaiveDataset,
}

def dataset_factory(cfg):
    dataset = DATASETS[cfg['dataset_name']]
    return dataset(cfg).data
