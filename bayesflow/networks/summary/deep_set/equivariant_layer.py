from collections.abc import Sequence

import keras
from keras import ops

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from ...subnets import MLP

from .invariant_layer import InvariantLayer


@serializable("bayesflow.networks")
class EquivariantLayer(keras.Layer):
    """Implements a permutation-equivariant transform with Pre-LN RMSNorm.

    Projects the input to ``embed_dim``, then applies:

        h = input_projector(x)
        h = h + equivariant_fc(concat([LN(h), tile(invariant_module(LN(h)))]))

    The residual works without an output projector because both the residual
    base ``h`` and the MLP output share ``embed_dim``.

    For details and justification, see:

    [1] Bloem-Reddy, B., & Teh, Y. W. (2020). Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1. https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """

    def __init__(
        self,
        embed_dim: int = 128,
        mlp_widths: Sequence[int] = (128,),
        pooling: str = "mean",
        activation: str = "gelu",
        kernel_initializer: str = "he_normal",
        dropout: float | None = 0.05,
        **kwargs,
    ):
        """
        Parameters
        ----------
        embed_dim : int, optional
            Working dimensionality. Input is projected to this width and the
            residual preserves it throughout. Default is 128.
        mlp_widths : Sequence[int], optional
            Hidden layer widths for the equivariant MLP and the internal
            invariant sub-MLPs. The MLP output is always ``embed_dim``.
            Default is ``(128,)``.
        pooling : str, optional
            Pooling used inside the invariant submodule, e.g. ``"mean"``.
            Default is ``"mean"``.
        activation : str, optional
            Activation function. Default is ``"gelu"``.
        kernel_initializer : str, optional
            Weight initializer. Default is ``"he_normal"``.
        dropout : float or None, optional
            Dropout rate. Default is 0.05.
        **kwargs
            Additional keyword arguments forwarded to ``keras.Layer``.
        """
        super().__init__(**layer_kwargs(kwargs))

        self.embed_dim = embed_dim
        self.mlp_widths = tuple(mlp_widths)
        self.pooling = pooling
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout

        # Project input to embed_dim so the residual connection has a fixed width
        self.input_projector = keras.layers.Dense(embed_dim, use_bias=False, kernel_initializer=kernel_initializer)

        self.ln = keras.layers.RMSNormalization()
        self.ln_fc = keras.layers.RMSNormalization()

        self.invariant_module = InvariantLayer(
            embed_dim=embed_dim,
            mlp_widths=mlp_widths,
            pooling=pooling,
            activation=activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
        )

        self.equivariant_fc = MLP(
            widths=(*mlp_widths, embed_dim),
            activation=activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
        )

    def call(self, x: Tensor, training: bool = False) -> Tensor:
        """Performs the forward pass of a learnable equivariant transform.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch_size, set_size, input_dim)``.
        training : bool, optional
            Passed to dropout and norm layers, by default False.

        Returns
        -------
        Tensor
            Output of shape ``(batch_size, set_size, embed_dim)``.
        """
        h = self.input_projector(x)

        h_norm = self.ln(h, training=training)
        inv = self.invariant_module(h_norm, training=training)

        inv_tiled = ops.tile(ops.expand_dims(inv, axis=-2), [1, ops.shape(h)[1], 1])
        h_cat = ops.concatenate([h_norm, inv_tiled], axis=-1)

        return h + self.equivariant_fc(self.ln_fc(h_cat, training=training), training=training)

    def build(self, input_shape):
        input_shape = tuple(input_shape)

        self.input_projector.build(input_shape)
        proj_shape = input_shape[:-1] + (self.embed_dim,)

        self.ln.build(proj_shape)
        self.invariant_module.build(proj_shape)

        # concat of [LN(h), tiled invariant]: (B, S, embed_dim + embed_dim)
        concat_shape = proj_shape[:-1] + (2 * self.embed_dim,)
        self.ln_fc.build(concat_shape)
        self.equivariant_fc.build(concat_shape)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)[:-1] + (self.embed_dim,)

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)
        return base_config | serialize(
            {
                "embed_dim": self.embed_dim,
                "mlp_widths": self.mlp_widths,
                "pooling": self.pooling,
                "activation": self.activation,
                "kernel_initializer": self.kernel_initializer,
                "dropout": self.dropout_rate,
            }
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
