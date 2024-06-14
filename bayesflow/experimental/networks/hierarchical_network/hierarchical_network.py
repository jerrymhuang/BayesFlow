
import keras

from keras.saving import (
    register_keras_serializable
)
from bayesflow.experimental.types import Tensor


@register_keras_serializable(package="bayesflow.networks.hierarchical_network")
class HierarchicalNetwork(keras.Model):
    """Implements a hierarchical summary network according to [1].

    [1] Elsemüller, L., Schnuerch, M., Bürkner, P. C., & Radev, S. T. (2023).
        A Deep Learning Method for Comparing Bayesian Hierarchical Models.
        arXiv preprint arXiv:2301.11873.
    """

    def __init__(self, networks_list, **kwargs):
        """Creates a hierarchical network consisting of stacked summary networks (one for each hierarchical level)
        that are aligned with the probabilistic structure of the processed data.

        Note: The networks will start processing from the lowest hierarchical level (e.g., observational level)
        up to the highest hierarchical level. It is recommended to provide higher-level networks with more
        expressive power to allow for an adequate compression of lower-level data.

        Example: For two-level hierarchical models with the assumption of temporal dependencies on the lowest
        hierarchical level (e.g., observational level) and exchangeable units at the higher level
        (e.g., group level), a list of [SequenceNetwork(), DeepSet()] could be passed.

        ----------

        Parameters:
        networks_list : list of keras.Model:
            The list of summary networks (one per hierarchical level), starting from the lowest hierarchical level
        """

        super().__init__(**kwargs)
        self.networks = networks_list

    def call(self, x: Tensor, return_all=False, **kwargs):
        """
        Performs the forward pass through the hierarchical network,
        transforming the nested input into learned summary statistics.

        Parameters
        ----------
        x          : KerasTensor of shape (batch_size, ..., data_dim)
            Example, hierarchical data sets with two levels:
            (batch_size, D, L, x_dim) -> reduces to (batch_size, out_dim).
        return_all : boolean, optional, default: False
            Whether to return all intermediate outputs (True) or just
            the final one (False).

        Returns
        -------
        out : tf.Tensor
            Output of shape ``(batch_size, out_dim) if return_all=False`` else a tuple
            of ``len(outputs) == len(networks)`` corresponding to all outputs of all networks.
        """

        if return_all:
            outputs = []
            for net in self.networks:
                x = net(x, **kwargs)
                outputs.append(x)
            return outputs
        else:
            for net in self.networks:
                x = net(x, **kwargs)
            return x
