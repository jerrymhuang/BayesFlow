
import keras


class ConditionalResidualBlock(keras.Model):
    """
    Implements a simple configurable MLP with optional residual connections and dropout.

    If used in conjunction with a coupling net, a diffusion model, or a flow matching model, it assumes
    that the input and conditions are already concatenated (i.e., this is a single-input model).
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim=512,
        num_hidden=2,
        activation="gelu",
        residual=True,
        spectral_norm=False,
        dropout_rate=0.05,
        zero_output_init=True,
        **kwargs,
    ):
        """
        Creates an instance of a flexible and simple MLP with optional residual connections and dropout.

        Parameters:
        -----------
        output_dim       : int
            The output dimensionality, needs to be specified according to the model's function.
        hidden_dim       : int, optional, default: 512
            The dimensionality of the hidden layers
        num_hidden       : int, optional, default: 2
            The number of hidden layers (minimum: 1)
        activation       : string, optional, default: 'gelu'
            The activation function of the dense layers
        residual         : bool, optional, default: True
            Use residual connections in the MLP.
        spectral_norm    : bool, optional, default: True
            Use spectral normalization for the network weights, which can make
            the learned function smoother and hence more robust to perturbations.
        dropout_rate     : float, optional, default: 0.05
            Dropout rate for the hidden layers in the MLP
        zero_output_init : bool, optional, default: True
            Will initialize the last layer's kernel to zeros, which can be helpful
            when used in conjunction with coupling layers.
        """

        super().__init__(**kwargs)

        self.dim = output_dim
        self.model = keras.Sequential()
        for _ in range(num_hidden):
            self.model.add(
                ConfigurableHiddenBlock(
                    num_units=hidden_dim,
                    activation=activation,
                    residual=residual,
                    dropout_rate=dropout_rate,
                    spectral_norm=spectral_norm
                )
            )
        if zero_output_init:
            output_initializer = "zeros"
        else:
            output_initializer = "glorot_uniform"
        self.output_layer = keras.layers.Dense(output_dim, kernel_initializer=output_initializer)

    def call(self, inputs, **kwargs):

        out = self.model(inputs, **kwargs)
        return self.output_layer(out)


class ConfigurableHiddenBlock(keras.layers.Layer):
    def __init__(
        self,
        num_units,
        activation="gelu",
        residual=True,
        dropout_rate=0.05,
        spectral_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.activation_fn = keras.activations.get(activation)
        self.residual = residual
        self.dense_with_dropout = keras.Sequential()
        if spectral_norm:
            self.dense_with_dropout.add(keras.layers.SpectralNormalization(keras.layers.Dense(num_units)))
        else:
            self.dense_with_dropout.add(keras.layers.Dense(num_units))
        self.dense_with_dropout.add(keras.layers.Dropout(dropout_rate))

    def call(self, inputs, **kwargs):
        x = self.dense_with_dropout(inputs, **kwargs)
        if self.residual:
            x = x + inputs
        return self.activation_fn(x)
