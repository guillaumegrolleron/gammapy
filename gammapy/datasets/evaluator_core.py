from gammapy.modeling.parameter import Parameters
from gammapy.modeling.models import TemplateNPredModel


class Evaluator:
    """Base class for sky model evaluation on maps.

    This class provides common functionality for evaluating sky models on maps,
    including handling of bias parameters.
    """

    def __init__(self, bias_parameters=None):
        self.bias_parameters = (
            bias_parameters if bias_parameters is not None else Parameters()
        )

    @property
    def bias_parameters_changed(self):
        """Check if any bias parameter has changed since last evaluation."""
        res = {}
        for param in self.bias_parameters:
            if (
                not hasattr(self, f"__bias_{param.name}_cached")
                or getattr(self, f"__bias_{param.name}_cached") != param.value
            ):
                res[param.name] = True
                setattr(self, f"__bias_{param.name}_cached", param.value)
            else:
                res[param.name] = False
        return res

    @property
    def bias_parameters(self) -> Parameters:
        return self._bias_parameters

    @bias_parameters.setter
    def bias_parameters(self, value: Parameters):
        self._bias_parameters = value
        for param in value._parameters:
            setattr(self, f"__bias_{param.name}_cached", param.value)

    @property
    def needs_update(self):
        """Check whether the model component has drifted away from its support."""
        if isinstance(self.model, TemplateNPredModel):
            return False
        elif not self.contributes:
            return False
        elif self.exposure is None:
            return True
        elif self.bias_parameters_changed.get("energy_reco_bias", False):
            return True
        elif self.geom.is_region:
            return False
        elif self.evaluation_mode == "global" or (
            hasattr(self.model, "evaluation_radius")
            and self.model.evaluation_radius is None
        ):
            return False
        elif not self.parameters_spatial_changed(reset=False):
            return False

        else:
            return self.irf_position_changed
