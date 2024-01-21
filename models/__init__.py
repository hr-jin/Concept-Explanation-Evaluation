from .pythia_70m import Pythia70M
from .llama_2_7b_chat import Llama2Chat7B

MODELS = {
    Pythia70M.code(): Pythia70M,
    Llama2Chat7B.code(): Llama2Chat7B,
}


def model_factory(cfg):
    model = MODELS[cfg['model_to_interpret']]
    return model(cfg).model
