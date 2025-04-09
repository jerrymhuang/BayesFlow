from collections.abc import Sequence
from typing import Literal

import keras

from bayesflow.types import Tensor
from bayesflow.utils import keras_kwargs
from bayesflow.utils.serialization import serializable
from .hidden_block import ConfigurableHiddenBlock


@serializable
class MLP(keras.Layer):
    """
    Implements a simple configurable MLP with optional residual connections and dropout.

    If used in conjunction with a coupling net, a diffusion model, or a flow matching model, it assumes
    that the input and conditions are already concatenated (i.e., this is a single-input model).
    """

    def __init__(
        self,
        widths: Sequence[int] = (256, 256),
        *,
        activation: str = "mish",
        kernel_initializer: str = "he_normal",
        residual: bool = False,
        dropout: Literal[0, None] | float = 0.05,
        spectral_normalization: bool = False,
        **kwargs,
    ):
        """
        Implements a flexible multi-layer perceptron (MLP) with optional residual connections, dropout, and
        spectral normalization.

        This MLP can be used as a general-purpose feature extractor or function approximator,supporting configurable
        depth, width, activation functions, and weight initializations.

        If `residual` is enabled, each layer includes a skip connection for improved gradient flow. The model also
        supports dropout for regularization and spectral normalization for stability in learning smooth functions.

        Parameters
        ----------
        widths : Sequence[int], optional
            Defines the number of hidden units per layer, as well as the number of layers to be used.
        activation : str, optional
            Activation function applied in the hidden layers, such as "mish". Default is "mish".
        kernel_initializer : str, optional
            Initialization strategy for kernel weights, such as "he_normal". Default is "he_normal".
        residual : bool, optional
            Whether to use residual connections for improved training stability. Default is False.
        dropout : float or None, optional
            Dropout rate applied within the MLP layers for regularization. Default is 0.05.
        spectral_normalization : bool, optional
            Whether to apply spectral normalization to stabilize training. Default is False.
        **kwargs
            Additional keyword arguments passed to the Keras layer initialization.
        """

        super().__init__(**keras_kwargs(kwargs))

        self.res_blocks = []
        for width in widths:
            self.res_blocks.append(
                ConfigurableHiddenBlock(
                    units=width,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    residual=residual,
                    dropout=dropout,
                    spectral_normalization=spectral_normalization,
                )
            )

    def build(self, input_shape):
        if self.built:
            # rebuilding when the network is already built can cause issues with serialization
            # see https://github.com/keras-team/keras/issues/21147
            return

        for layer in self.res_blocks:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)

    def call(self, x: Tensor, training: bool = False, **kwargs) -> Tensor:
        for layer in self.res_blocks:
            x = layer(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        for layer in self.res_blocks:
            input_shape = layer.compute_output_shape(input_shape)

        return input_shape
