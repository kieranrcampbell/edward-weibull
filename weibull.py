from edward.models import RandomVariable
import tensorflow as tf
from tensorflow.contrib.distributions import Distribution


class Weibull(RandomVariable, Distribution):
    """Weibull distribution

    The Weibull distribution is defined over the non-negative real numbers.

    Here we use the medical statistics parameterisation with
    shape parameter k and scale parameter p. This is closely related
    to the parameterisation used in numpy with lambda = -k log(p).

    The probability density function is

    pdf(x; k > 0, b > 0) = bkx^{k-1} e^{-bx^k}
    """

    def __init__(self, shape, scale, validate_args = False, allow_nan_stats = True,
               name = "weibull", *args, **kwargs):
        """Initialise a Weibull random variable

        Parameters
        -------
        shape: tf.Tensor
        scale: tf.Tensor
        -------
        """
        # self.k = tf.identity(shape, name = "k")
        # self.p = tf.identity(scale, name = "p")

        # parameters = locals()
        # parameters.pop("self")
        parameters = {'shape': shape, 'scale': scale}



        with tf.name_scope(name, values=[shape, scale]):
            with tf.control_dependencies([
                tf.assert_positive(shape),
                tf.assert_positive(scale),
            ] if validate_args else []):
                self._shape = tf.identity(
                    shape, name="shape")
                self._scale = tf.identity(scale, name="scale")
                # contrib_tensor_util.assert_same_float_dtype(
                    # [self._shape, self._scale])

        super(Weibull, self).__init__(
            dtype = self._shape.dtype,
            validate_args = validate_args,
            allow_nan_stats = allow_nan_stats,
            is_continuous = True,
            is_reparameterized = False,
            parameters=parameters,
            graph_parents=[self._shape, self._scale],
            name = name,
            *args, **kwargs)

        self._kwargs = parameters

    def _batch_shape(self):
        return tf.broadcast_static_shape(
            self._shape.get_shape(),
            self._scale.get_shape())

    def _get_event_shape(self):
        # change to just tensorshape
        return tf.placeholder(tf.float32, shape = []).get_shape()

    def _log_prob(self, value):
        lp = tf.log(self._scale) + tf.log(self._shape) + \
        (self._shape - 1) * tf.log(value) - self._scale * tf.pow(value, self._shape)
        return lp

    def _sample_n(self, n, seed=None):
        ## Need to convert to (k, lambda) rep
        batch_shape = tf.convert_to_tensor(self._batch_shape(), dtype = "int32")
        n_tens = tf.convert_to_tensor([n], dtype = "int32")

        U = tf.random_uniform(tf.concat([n_tens, batch_shape], axis = 0))
        X = tf.pow(-tf.log(U) / self._scale, 1 / self._shape)
        return X


class TruncatedWeibull(RandomVariable, Distribution):
    """Truncated Weibull distribution

    The Weibull distribution is defined over the non-negative real numbers.

    Here we use the medical statistics parameterisation with
    shape parameter k and scale parameter b. This is closely related
    to the parameterisation used in numpy with lambda = -k log(b).

    The probability density function is

    pdf(x; k > 0, b > 0, o) = bkx**{k-1} e**{-bx^k} / S(o; k, b)

    where S is the Survival function which acts as the normalising constant,
    defined as

    S(x; k, b) = exp(-k x**b)
    """

    def __init__(self, shape, scale, cens, validate_args = False, allow_nan_stats = True,
               name = "truncated_weibull", *args, **kwargs):
        """Initialise a Truncated Weibull random variable

        Parameters
        -------
        shape: tf.Tensor
        scale: tf.Tensor
        -------
        """


        parameters = {'shape': shape, 'scale': scale, 'cens': cens}

        with tf.name_scope(name, values=[shape, scale]):
            with tf.control_dependencies([
                tf.assert_positive(shape),
                tf.assert_positive(scale)
            ] if validate_args else []):
                self._shape = tf.identity(
                    shape, name="shape")
                self._scale = tf.identity(scale, name="scale")
                self._cens = tf.identity(cens, name = "cens")

        super(TruncatedWeibull, self).__init__(
            dtype=self._shape.dtype,
            validate_args = validate_args,
            allow_nan_stats = allow_nan_stats,
            is_continuous = True,
            is_reparameterized = False,
            parameters=parameters,
            graph_parents=[self._shape, self._scale, self._cens],
            name = name,
            *args, **kwargs)
        self._kwargs = parameters

    def _log_S(self, x):
        return -self._scale * tf.pow(x, self._shape)

    def _log_prob(self, value):
        lp = tf.log(self._scale) + tf.log(self._shape) + \
        (self._shape - 1) * tf.log(value) - self._scale * tf.pow(value, self._shape)
        return lp - self._log_S(self._cens)

    def _batch_shape(self):
        return tf.broadcast_static_shape(
            self._shape.get_shape(),
            self._scale.get_shape())

    def _get_event_shape(self):
        # change to just tensorshape
        return tf.placeholder(tf.float32, shape = []).get_shape()

    def _phi(self, x):
        """Cumulative distribution function """
        return 1 - tf.exp(self._log_S(x))

    def _sample_n(self, n, seed=None):
        batch_shape = tf.convert_to_tensor(self._batch_shape(), dtype = "int32")
        n_tens = tf.convert_to_tensor([n], dtype = "int32")

        lower_bound = self._phi(self._cens)

        U = tf.random_uniform(tf.concat([n_tens, batch_shape], axis = 0))

        Ul = lower_bound + U * (1 - lower_bound) # constrain U \in (a, 1)

        X = tf.pow(-tf.log(1 - Ul) / self._scale, 1 / self._shape)
        return X



class TruncatedWeibull2(RandomVariable, Distribution):
    """Truncated Weibull distribution

    The Weibull distribution is defined over the non-negative real numbers.

    Here we use the medical statistics parameterisation with
    shape parameter k and scale parameter p. This is closely related
    to the parameterisation used in numpy with lambda = -k log(p).

    The probability density function is

    pdf(x; k > 0, b > 0, o) = bkx**{k-1} e**{-bx^k} / S(o; k, b)

    where S is the Survival function which acts as the normalising constant,
    defined as

    S(x; k, b) = exp(-k x**b)
    """

    def __init__(self, shape, scale, validate_args = False, allow_nan_stats = True,
               name = "truncated_weibull", *args, **kwargs):
        """Initialise a Truncated Weibull random variable

        Parameters
        -------
        shape: tf.Tensor
        scale: tf.Tensor
        -------
        """

        self._cens = tf.zeros(tf.shape(shape))
        parameters = {'shape': shape, 'scale': scale}

        with tf.name_scope(name, values=[shape, scale]):
            with tf.control_dependencies([
                tf.assert_positive(shape),
                tf.assert_positive(scale)
            ] if validate_args else []):
                self._shape = tf.identity(
                    shape, name="shape")
                self._scale = tf.identity(scale, name="scale")

        super(TruncatedWeibull2, self).__init__(
            dtype=self._shape.dtype,
            validate_args = validate_args,
            allow_nan_stats = allow_nan_stats,
            is_continuous = True,
            is_reparameterized = False,
            parameters=parameters,
            graph_parents=[self._shape, self._scale],
            name = name,
            *args, **kwargs)
        self._kwargs = parameters

    def _log_S(self, x):
        return -self._scale * tf.pow(x, self._shape)

    def _log_prob(self, value):
        lp = tf.log(self._scale) + tf.log(self._shape) + \
        (self._shape - 1) * tf.log(value) - self._scale * tf.pow(value, self._shape)
        return lp - self._log_S(self._cens)

    def _batch_shape(self):
        return tf.broadcast_static_shape(
            self._shape.get_shape(),
            self._scale.get_shape())

    def _get_event_shape(self):
        # change to just tensorshape
        return tf.placeholder(tf.float32, shape = []).get_shape()

    def _phi(self, x):
        """Cumulative distribution function """
        return 1 - tf.exp(self._log_S(x))

    def _sample_n(self, n, seed=None):
        batch_shape = tf.convert_to_tensor(self._batch_shape(), dtype = "int32")
        n_tens = tf.convert_to_tensor([n], dtype = "int32")

        lower_bound = self._phi(self._cens)

        U = tf.random_uniform(tf.concat([n_tens, batch_shape], axis = 0))

        Ul = lower_bound + U * (1 - lower_bound) # constrain U \in (a, 1)

        X = tf.pow(-tf.log(1 - Ul) / self._scale, 1 / self._shape)
        return X
