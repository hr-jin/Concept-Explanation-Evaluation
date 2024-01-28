from typing import Optional,Iterable
import torch
from uncertainties import ufloat
import math
from ud_treebank_utils.models import ProbingModel, ValueModel
from ud_treebank_utils.reader import Reader


class Estimator:
    """
    Base class for an MI estimator.
    """
    def __init__(self, probing_model: ProbingModel, value_model: Optional[ValueModel] = None):
        self._probing_model = probing_model
        self._value_model = value_model or self._probing_model.get_value_model()

    def estimate_integral(self, value_name: str) -> ufloat:
        """
        Estimates the integral we need to compute
        """
        raise NotImplementedError

    def estimate_conditional_entropy(self) -> ufloat:
        conditional_prob = ufloat(0.0, 0.0)
        with torch.no_grad():
            for v_name, val_prob in zip(self._value_model.get_values(), self._value_model.get_probs()):
                integral_estimate = self.estimate_integral(v_name)
                local_conditional_prob = -val_prob * integral_estimate
                conditional_prob += local_conditional_prob

        return conditional_prob


class FixedSamplesEstimator(Estimator):
    def __init__(self, probing_model: ProbingModel, reader: Reader, attribute: str,
                 select_dimensions: Iterable[int], value_model: ValueModel):
        self._reader = reader
        self._attribute = attribute
        self._select_dimensions = list(select_dimensions)

        super().__init__(probing_model, value_model=value_model)

    def estimate_integral(self, value_name: str) -> ufloat:
        """
        Estimates the integral we need to compute
        """
        # Let samples be the ones we have
        value = self._probing_model._value_model.get_value_ids([value_name]).cpu().tolist()[0]
        filter = lambda x: x.has_attribute(self._attribute) and x.get_attribute(self._attribute) == value_name
        embeddings = self._reader.get_embeddings_with_filter_from_cache(
            f"{value_name}_{value}", filter)[:, self._select_dimensions]
        num_samples = embeddings.shape[0]

        attribute_values = value * torch.ones(num_samples).to(self._probing_model.get_device())
        model_samples = embeddings.to(self._probing_model.get_device())

        # Compute probabilities
        log_prob = self._probing_model.log_prob_conditional(model_samples, attribute_values)
        log_prob_normalizer = self._probing_model.log_prob(model_samples)

        sampled_log_prob = log_prob - log_prob_normalizer
        mean_log_prob = sampled_log_prob.mean().item()
        std_log_prob = sampled_log_prob.std().item() / math.sqrt(num_samples)

        # Return estimated mean with 95% confidence bound
        return ufloat(mean_log_prob, 2 * std_log_prob)
