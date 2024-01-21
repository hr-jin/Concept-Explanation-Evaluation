from .pile_tokenized_10b import PileTokenized10BDataset
from .harmful_qa import HarmFulQADataset
from .truthful_qa import TrustfulQADataset

DATASETS = {
    PileTokenized10BDataset.code(): PileTokenized10BDataset,
    HarmFulQADataset.code(): HarmFulQADataset,
    TrustfulQADataset.code(): TrustfulQADataset,
}

def dataset_factory(cfg):
    dataset = DATASETS[cfg['dataset_name']]
    return dataset(cfg).data
