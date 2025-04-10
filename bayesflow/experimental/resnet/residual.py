import keras

from bayesflow.types import Tensor
from bayesflow.utils import sequential_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize


@serializable
class Residual(keras.Sequential):
    def __init__(self, *layers: keras.Layer, **kwargs):
        super().__init__(list(layers), **sequential_kwargs(kwargs))
        self.projector = keras.layers.Dense(units=None)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()

        config = {
            "layers": self.layers,
            "projector": self.projector,
        }

        return base_config | serialize(config)

    def build(self, input_shape=None):
        if self.built:
            return

        super().build(input_shape)
        output_shape = super().compute_output_shape(input_shape)

        self.projector.units = input_shape[-1]
        self.projector.build(output_shape)

        self.built = True

    def call(self, x: Tensor, training: bool = None, mask: bool = None) -> Tensor:
        return x + self.projector(super().call(x, training=training, mask=mask))


# @serializable
# class Residual(keras.Model):
#     def __init__(self, inner: keras.Layer, **kwargs):
#         super().__init__(**model_kwargs(kwargs))
#         self.inner = inner
#         self.projector = keras.layers.Dense(units=None)
#
#     @classmethod
#     def from_config(cls, config, custom_objects=None):
#         return cls(**deserialize(config, custom_objects=custom_objects))
#
#     def get_config(self):
#         base_config = super().get_config()
#
#         config = {
#             "layer": self.inner,
#             "projector": self.projector,
#         }
#
#         return base_config | serialize(config)
#
#     def build(self, input_shape=None):
#         if self.built:
#             return
#
#         self.inner.build(input_shape)
#         output_shape = self.inner.compute_output_shape(input_shape)
#
#         self.projector.units = input_shape[-1]
#         self.projector.build(output_shape)
#
#         self.built = True
#
#     def call(self, x: Tensor, training: bool = None, mask: bool = None) -> Tensor:
#         return x + self.projector(self.inner(x, training=training, mask=mask))
