import keras
from keras import layers
from keras.saving import register_keras_serializable

from bayesflow.types import Tensor


@register_keras_serializable(package="bayesflow.networks")
class ConfigurableHiddenBlock(keras.layers.Layer):
    def __init__(
        self,
        units: int = 256,
        activation: str = "mish",
        kernel_initializer: str = "he_normal",
        residual: bool = True,
        dropout: float = 0.05,
        spectral_normalization: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.activation_fn = keras.activations.get(activation)
        self.residual = residual
        self.dense = layers.Dense(
            units=units,
            kernel_initializer=kernel_initializer,
        )
        if spectral_normalization:
            self.dense = layers.SpectralNormalization(self.dense)
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, inputs: Tensor, training=False):
        x = self.dense(inputs, training=training)
        x = self.dropout(x, training=training)
        if self.residual:
            x = x + inputs
        return self.activation_fn(x)

    def build(self, input_shape):
        # build nested layers with forward pass
        self.call(keras.ops.zeros(input_shape))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "residual": self.residual,
                "activation_fn": keras.saving.serialize_keras_object(self.activation_fn),
                "dense": keras.saving.serialize_keras_object(self.dense),
                "dropout": keras.saving.serialize_keras_object(self.dropout),
            }
        )
        return config