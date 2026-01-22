from gammapy.makers import Maker
from gammapy.modeling.parameter import Parameter

__all__ = [
    "IRFBiasMaker",
]


class IRFBiasMaker(Maker):
    """Make event dataset for a single IACT observation."""

    tag = "IRFBiasMaker"

    def __init__(
        self,
        energy_reco_bias: Parameter = Parameter("energy_reco_bias", 1.0, frozen=True),
        background_bias: Parameter = Parameter("background_bias", 1.0, frozen=True),
    ):
        self.energy_reco_bias = energy_reco_bias
        self.background_bias = background_bias

    def run(self, dataset, observation):
        dataset.energy_reco_bias = self.energy_reco_bias.copy()
        dataset.background_bias = self.background_bias.copy()
        return dataset
