from typing import List, Tuple, Optional
import numpy as np
from ud_treebank_utils.word import Word
from typing import Tuple, List
import numpy as np
from overrides import overrides
from tqdm import trange
import torch
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal


class Trainer:
    def compute_gaussian_model_params_for_word_list(self, words: List[Word]) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def compute_gaussian_model_params_for_attribute_value(
            self, attribute: str, value: str, words: List[Word]) -> Tuple[np.ndarray, np.ndarray]:
        """
        This is used to extract the gaussian model parameters for an attribute-value pair.

        aka. mean and cov for p(h|v,a)
        """
        filtered_words = [x for x in words if x.has_attribute(attribute) and x.get_attribute(attribute) == value]

        return self.compute_gaussian_model_params_for_word_list(filtered_words)

    def compute_categorical_model_sampling_prob_for_attribute_value(
            self, attribute: str, value: str, words: List[Word]) -> float:
        """
        This is used to extract the probability of sampling a specific attribute-value pair.

        aka. p(v|a)

        In practice this will be estimated from a large corpus.
        """
        words_with_attribute = float(sum([x.get_count() for x in words if x.has_attribute(attribute)]))
        words_with_attribute_value = float(
            sum([x.get_count() for x in words
                if x.has_attribute(attribute) and x.get_attribute(attribute) == value]))

        return words_with_attribute_value / words_with_attribute


class MLETrainer(Trainer):
    @overrides
    def compute_gaussian_model_params_for_word_list(self, words: List[Word]) -> Tuple[np.ndarray, np.ndarray]:
        embeddings = [np.array(x.get_embedding()) for x in words]

        # Create numpy array where rows are embeddings
        embeddings_array = np.vstack(embeddings)

        # Compute mean and covariance
        mean = np.mean(embeddings_array, axis=0)
        cov = np.cov(embeddings_array, rowvar=False)

        return mean, cov
    

class MAPTrainer(MLETrainer):
    """
    Learns gaussians using Maximum a Posteriori. This means that instead of within a set of parameters `w` s.t.

    w = argmax_w p(D|w)

    we find

    w = argmax_w p(w|D) = argmax_w p(D|w) p(w)
    """
    def __init__(self, device: torch.device,
                 # Parameters mean
                 mu: np.array, k: float,
                 # Parameters covariance
                 nu: float, Lambda: np.array,
                 # If this is set, overrides all arguments passed and uses a data-dependent prior
                 # from the literature
                 initialize_from_data: bool = False):
        self.device = device

        self.mu = mu
        self.k = k
        self.nu = nu
        self.Lambda = Lambda

        self.initialize_from_data = initialize_from_data

    @classmethod
    def from_data(cls, device: torch.device):
        dimension = 42
        mu = torch.zeros(dimension)
        return cls(
            device=device, initialize_from_data=True,
            mu=mu, k=1.0, nu=float(dimension), Lambda=torch.eye(dimension))

    @classmethod
    def from_dimension(cls, device: torch.device, dimension: int, mu: Optional[torch.tensor] = None, k: float = 1.0):
        mu = mu or torch.zeros(dimension)
        return cls(device=device, mu=mu, k=k, nu=float(dimension), Lambda=np.eye(dimension))

    @overrides
    def compute_gaussian_model_params_for_word_list(self, words: List[Word]) -> Tuple[np.ndarray, np.ndarray]:
        embeddings = [torch.tensor(x.get_embedding()) for x in words]

        # Create numpy array where rows are embeddings
        embeddings_array = torch.stack(embeddings, dim=0).to(self.device)
        embeddings_mean = torch.mean(embeddings_array, dim=0)
        embeddings_scatter = (embeddings_array - embeddings_mean).t().matmul(embeddings_array - embeddings_mean)

        d = embeddings_mean.shape[0]
        n = len(words)

        if self.initialize_from_data:
            centered_embeddings = (embeddings_array - embeddings_mean)
            cov = centered_embeddings.t().matmul(centered_embeddings) / n

            self.Lambda = torch.diag(torch.diag(cov))
            self.nu = d + 2
            self.mu = embeddings_mean
            self.k = 0.01

        mu_update = (self.k * self.mu + n * embeddings_mean) / (self.k + n)

        k_update = self.k + n
        nu_update = self.nu + n

        embeddings_scatter_prior = (embeddings_mean - self.mu) @ (embeddings_mean - self.mu).T
        Lambda_update = self.Lambda + embeddings_scatter
        Lambda_update += (self.k * n) / (self.k + n) * embeddings_scatter_prior

        # Get MAP estimates for mean and covariance
        mean = mu_update
        cov = (nu_update + d + 2) ** -1 * Lambda_update
        return mean.cpu().numpy(), cov.cpu().numpy()


