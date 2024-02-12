from .pile_tokenized_10b import PileTokenized10BDataset
from .harmful_qa import HarmFulQADataset
from .truthful_qa import TrustfulQADataset
from .conceptx_data import ConceptXData
from .conceptx_naive import ConceptXNaiveDataset

DATASETS = {
    PileTokenized10BDataset.code(): PileTokenized10BDataset,
    HarmFulQADataset.code(): HarmFulQADataset,
    TrustfulQADataset.code(): TrustfulQADataset,
    ConceptXData.code(): ConceptXData,
    ConceptXNaiveDataset.code(): ConceptXNaiveDataset,
}

def dataset_factory(cfg):
    dataset = DATASETS[cfg['dataset_name']]
    return dataset(cfg).data
