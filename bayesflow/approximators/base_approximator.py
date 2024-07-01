import keras
from keras.saving import (
    deserialize_keras_object,
    register_keras_serializable,
    serialize_keras_object,
)
from collections.abc import Sequence
import warnings

from bayesflow.configurators import BaseConfigurator
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.types import Shape, Tensor
from bayesflow.utils import keras_kwargs, repeat_tensor, process_output


@register_keras_serializable(package="bayesflow.approximators")
class BaseApproximator(keras.Model):
    def __init__(
        self,
        inference_network: InferenceNetwork,
        summary_network: SummaryNetwork,
        configurator: BaseConfigurator,
        **kwargs,
    ):
        super().__init__(**keras_kwargs(kwargs))
        self.inference_network = inference_network
        self.summary_network = summary_network
        self.configurator = configurator

    def sample(self, data: dict[str, Tensor], num_samples: int = 500, as_numpy: bool = True) -> Tensor:
        """Generates ``num_samples'' from the approximate distribution. Will typically be called only on
        trained models.

        Parameters
        ----------
        data: dict[str: Tensor]
            The data dictionary containing all keys used when constructing the Approximator except
            ``inference_variables'', which is assumed to be absent during inference and will be ignored
            if present.
        num_samples: int, optional, default - 500
            The number of samples per data set / instance in the data dictionary.
        as_numpy: bool, optional, default - True
            An optional flag to convert the samples to a numpy array before returning.

        Returns
        -------
        samples: Tensor
            A tensor of shape (num_data_sets, num_samples, num_inference_variables) if data contains
            multiple data sets / instances or of shape (num_samples, num_inference_variables) if data
            contains a single data sets (i.e., a leading axis with one element in the corresponding
            conditioning variables).
        """

        data = data.copy()

        if self.summary_network is None:
            data["inference_conditions"] = self.configurator.configure_inference_conditions(data)

        else:
            data["summary_conditions"] = self.configurator.configure_summary_conditions(data)
            data["summary_variables"] = self.configurator.configure_summary_variables(data)
            summary_metrics = self.summary_network.compute_metrics(data, stage="inference")
            data["summary_outputs"] = summary_metrics.get("outputs")

            data["inference_conditions"] = self.configurator.configure_inference_conditions(data)

        data["inference_conditions"] = repeat_tensor(data["inference_conditions"], num_repeats=num_samples, axis=1)
        samples = self.inference_network.sample(num_samples, data["inference_conditions"])

        return process_output(samples, convert_to_numpy=as_numpy)

    def log_prob(self, data: dict[str, Tensor], as_numpy: bool = True) -> Tensor:
        """TODO"""

        data = data.copy()

        if self.summary_network is None:
            data["inference_conditions"] = self.configurator.configure_inference_conditions(data)

        else:
            data["summary_conditions"] = self.configurator.configure_summary_conditions(data)
            data["summary_variables"] = self.configurator.configure_summary_variables(data)
            summary_metrics = self.summary_network.compute_metrics(data, stage="inference")
            data["summary_outputs"] = summary_metrics.get("outputs")

            data["inference_conditions"] = self.configurator.configure_inference_conditions(data)

        data["inference_variables"] = self.configurator.configure_inference_variables(data)
        log_density = self.inference_network.log_prob(data["inference_variables"], data["inference_conditions"])

        return process_output(log_density, convert_to_numpy=as_numpy)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "BaseApproximator":
        config["inference_network"] = deserialize_keras_object(
            config["inference_network"], custom_objects=custom_objects
        )
        config["summary_network"] = deserialize_keras_object(config["summary_network"], custom_objects=custom_objects)
        config["configurator"] = deserialize_keras_object(config["configurator"], custom_objects=custom_objects)

        return cls(**config)

    def get_config(self) -> dict:
        base_config = super().get_config()

        config = {
            "inference_network": serialize_keras_object(self.inference_network),
            "summary_network": serialize_keras_object(self.summary_network),
            "configurator": serialize_keras_object(self.configurator),
        }

        return base_config | config

    # noinspect PyMethodOverriding
    def build(self, data_shapes: dict[str, Shape]):
        data = {name: keras.ops.zeros(shape) for name, shape in data_shapes.items()}
        self.build_from_data(data)

    def build_from_data(self, data: dict[str, Tensor]):
        self.compute_metrics(data, stage="training")
        self.built = True

    def train_step(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        # we cannot provide a backend-agnostic implementation due to reliance on autograd
        raise NotImplementedError

    def test_step(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        metrics = self.compute_metrics(data, stage="validation")
        self._loss_tracker.update_state(metrics["loss"])
        return metrics

    def evaluate(self, *args, **kwargs):
        val_logs = super().evaluate(*args, **kwargs)

        if val_logs is None:
            # https://github.com/keras-team/keras/issues/19835
            warnings.warn(
                "Found no validation logs due to a bug in keras. "
                "Applying workaround, but incorrect loss values may be logged. "
                "If possible, increase the size of your dataset, "
                "or lower the number of validation steps used."
            )

            val_logs = {}

        return val_logs

    # noinspection PyMethodOverriding
    def compute_metrics(self, data: dict[str, Tensor], stage: str = "training") -> dict[str, Tensor]:
        # compiled modes do not allow in-place operations on the data object
        # we perform a shallow copy here, which is cheap
        data = data.copy()

        if self.summary_network is None:
            data["inference_variables"] = self.configurator.configure_inference_variables(data)
            data["inference_conditions"] = self.configurator.configure_inference_conditions(data)
            return self.inference_network.compute_metrics(data, stage=stage)

        data["summary_variables"] = self.configurator.configure_summary_variables(data)
        data["summary_conditions"] = self.configurator.configure_summary_conditions(data)

        summary_metrics = self.summary_network.compute_metrics(data, stage=stage)

        data["summary_outputs"] = summary_metrics.pop("outputs")

        data["inference_variables"] = self.configurator.configure_inference_variables(data)
        data["inference_conditions"] = self.configurator.configure_inference_conditions(data)

        inference_metrics = self.inference_network.compute_metrics(data, stage=stage)

        metrics = {"loss": summary_metrics["loss"] + inference_metrics["loss"]}

        summary_metrics = {f"summary/{key}": val for key, val in summary_metrics.items()}
        inference_metrics = {f"inference/{key}": val for key, val in inference_metrics.items()}

        return metrics | summary_metrics | inference_metrics

    def compute_loss(self, *args, **kwargs):
        raise RuntimeError("Use compute_metrics()['loss'] instead.")

    def fit(self, *args, **kwargs):
        if not self.built:
            try:
                dataset = kwargs.get("x") or args[0]
                data = next(iter(dataset))
                self.build_from_data(data)
            except Exception:
                raise RuntimeError(
                    "Could not automatically build the approximator. Please pass a dataset as the "
                    "first argument to `approximator.fit()` or manually call `approximator.build()` "
                    "with a dictionary specifying your data shapes."
                )

        return super().fit(*args, **kwargs)

    def compile(
        self, inference_metrics: Sequence[keras.Metric] = None, summary_metrics: Sequence[keras.Metric] = None, **kwargs
    ) -> None:
        self.inference_network._metrics = inference_metrics or []

        if self.summary_network is not None:
            self.summary_network._metrics = summary_metrics or []

        return super().compile(**kwargs)
