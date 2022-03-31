import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.layers import Activation
from tensorflow.keras.backend import sigmoid, constant
from tensorflow.keras.initializers import Initializer


class Swish(Activation):
    """
    Custom Swish activation function for Keras.
    """

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'Swish'


def swish1(x):
    """
    Standard Swish activation.

    Args:
        x: Keras tensor
            Input tensor

    Returns:
        Output tensor of Swish transformation.
    """

    return x * sigmoid(x)


def eswish(x):
    """
    E-swish activation with Beta value of 1.25.

    Args:
        x: Keras tensor
            Input tensor

    Returns:
        Output tensor of E-swish transformation.
    """

    beta = 1.25
    return beta * x * sigmoid(x)


class keras_BilinearWeights(Initializer):
    """
    A Keras implementation of bilinear weights by Joel Kronander (https://github.com/tensorlayer/tensorlayer/issues/53)
    """

    def __init__(self, shape=None, dtype=None):
        self.shape = shape
        self.dtype = dtype

    def __call__(self, shape=None, dtype=None):

        # Initialize parameters
        if shape:
            self.shape = shape
        self.dtype = type=np.float32 # Overwrites argument

        scale = 2
        filter_size = self.shape[0]
        num_channels = self.shape[2]

        # Create bilinear weights
        bilinear_kernel = np.zeros([filter_size, filter_size], dtype=self.dtype)
        scale_factor = (filter_size + 1) // 2
        if filter_size % 2 == 1:
            center = scale_factor - 1
        else:
            center = scale_factor - 0.5
        for x in range(filter_size):
            for y in range(filter_size):
                bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                                       (1 - abs(y - center) / scale_factor)

        # Assign weights
        weights = np.zeros((filter_size, filter_size, num_channels, num_channels))
        for i in range(num_channels):
            weights[:, :, i, i] = bilinear_kernel

        return constant(value=weights)

    def get_config(self):
        return {'shape': self.shape}

