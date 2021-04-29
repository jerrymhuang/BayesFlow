import tensorflow as tf


class MetaAmortizer(tf.keras.Model):

    def __init__(self, inference_net=None, evidence_net=None, summary_net=None):
        """ 
        Connects an evidential network with a summary network as in the BayesFlow for model comparison set-up.

        Parameters
        ----------
        inference_net : tf.keras.Model
            An (invertible) inference network which processes the outputs of a generative model (i.e., params, sim_data)
        evidence_net  : tf.keras.Model
            An evidential network which processes the outputs of multiple generative models (i.e., sim_data)
        summary_net   : tf.keras.Model or None, optional, default: None
            An optional summary network
        """
        super(MetaAmortizer, self).__init__()

        self.inference_net = inference_net
        self.evidence_net = evidence_net
        self.summary_net = summary_net

    def call(self, model_indices, params, sim_data):
        """ 
        Performs a forward pass through the networks.

        Paramters
        ---------
        model_indices  : tf.Tensor or np.array of shape (n_sim, n_models)
            the true, one-hot-encoded model indices :math:`m \sim p(m)`
        params         : tf.Tensor or np.array of shape (n_sim, n_params)
            the parameters :math:`\\theta \sim p(\\theta | m)` of interest
        sim_data       : tf.Tensor or np.array of shape (n_sim, n_obs, data_dim)
            the conditional data `x`

        Returns
        -------
        out_inference: np.array
            The output of the inference network or ``None`` if no networks provided
        out_evidence: np.array
            The output of the evidence network or ``None`` if no networks provided

        """

        # TODO - Model aware or model-agnostic summary
        if self.summary_net is not None:
            sim_data = self.summary_net(sim_data, model_indices)

        if self.evidence_net is not None:
            out_evidence = self.evidence_net(sim_data)
        else:
            out_evidence = None

        if self.inference_net is not None:
            out_inference = self.inference_net(model_indices, params, sim_data)
        else:
            out_inference = None
        return out_inference, out_evidence

    def sample_from_model(self, x, m_idx, n_samples, to_numpy=True):
        """Performs fast parallelized inference on a single model.
        """

        raise NotImplementedError('TODO!')

    def compare_models(self):
        """Performs model comparison.
        """

        raise NotImplementedError('TODO!')


class MultiModelAmortizer(tf.keras.Model):
    """ Connects an evidential network with a summary network as in the BayesFlow for model comparison set-up.
    """

    def __init__(self, evidence_net, summary_net=None):
        """Initializes a MultiModelAmortizer for

        Parameters
        ----------
        evidence_net : tf.keras.Model
            An evidential network which processes the outputs of multiple generative models (i.e., sim_data)
        summary_net   : tf.keras.Model or None, optional, default: None
            An optional summary network
        """
        super(MultiModelAmortizer, self).__init__()

        self.evidence_net = evidence_net
        self.summary_net = summary_net

    def call(self, sim_data):
        """Performs a forward pass through the summary and inference network.

        Parameters
        ----------
        sim_data  : tf.Tensor of shape (batch_size, n_obs, data_dim)
            The conditional data `x`

        Returns
        -------
        out : np.array
            The outputs of ``evidence_net(summary_net(x))``, usually model probabilities or absolute evidences
        """

        # Compute learnable summaries, if given
        if self.summary_net is not None:
            sim_data = self.summary_net(sim_data)

        # Compute output of inference net
        out = self.evidence_net(sim_data)
        return out

    def sample(self, obs_data, n_samples, **kwargs):
        """Performs inference on actually observed or simulated validation data.

        Parameters
        ----------
        obs_data  : tf.Tensor of shape (n_datasets, n_obs, data_dim)
            The conditional data set(s)
        n_samples : int
            The number of posterior samples to obtain from the approximate posterior

        Returns
        -------
        post_samples : tf.Tensor of shape (n_samples, n_datasets, n_models)
            The sampled model indices or evidences per dataset or model
        """

        # Compute learnable summaries, if given
        if self.summary_net is not None:
            obs_data = self.summary_net(obs_data)

        post_samples = self.evidence_net.sample(obs_data, n_samples, **kwargs)
        return post_samples


class SingleModelAmortizer(tf.keras.Model):
    """ Connects an inference network for parameter estimation with an optional summary network
    as in the original BayesFlow set-up.
    """
    def __init__(self, inference_net, summary_net=None):
        """Initializes the SingleModelAmortizer

        Parameters
        ----------
        inference_net : tf.keras.Model
            An (invertible) inference network which processes the outputs of a generative model (i.e., params, sim_data)
        summary_net   : tf.keras.Model or None, optional, default: None
            An optional summary network
        """
        super(SingleModelAmortizer, self).__init__()

        self.inference_net = inference_net
        self.summary_net = summary_net

    def call(self, params, sim_data):
        """ Performs a forward pass through the summary and inference network.

        Parameters
        ----------
        params    : tf.Tensor of shape (batch_size, n_params)
            the parameters theta ~ p(theta | x) of interest
        sim_data  : tf.Tensor of shape (batch_size, n_obs, data_dim)
            the conditional data x

        Returns
        -------
        out
            the outputs of ``inference_net(theta, summary_net(x))``, usually a latent variable and
            log(det(Jacobian)), that is a tuple ``(z, log_det_J)``
        """

        # Compute learnable summaries, if given
        if self.summary_net is not None:
            sim_data = self.summary_net(sim_data)

        # Compute output of inference net
        out = self.inference_net(params, sim_data)
        return out

    def sample(self, obs_data, n_samples, **kwargs):
        """ Performs inference on actually observed or simulated validation data.


        Parameters
        ----------

        obs_data  : tf.Tensor of shape (n_datasets, n_obs, data_dim)
            The conditional data set(s)
        n_samples : int
            The number of posterior samples to obtain from the approximate posterior

        Returns
        -------
        post_samples : tf.Tensor of shape (n_samples, n_datasets, n_params)
            the sampled parameters per data set
        """

        # Compute learnable summaries, if given
        if self.summary_net is not None:
            obs_data = self.summary_net(obs_data)

        post_samples = self.inference_net.sample(obs_data, n_samples, **kwargs)
        return post_samples
