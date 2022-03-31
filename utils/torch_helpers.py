import numpy as np
from torch.nn import ConvTranspose2d, init
from torch import Tensor


class pytorch_BilinearConvTranspose2d(ConvTranspose2d):
    """
    A PyTorch implementation of transposed bilinear convolution by mjstevens777 (https://gist.github.com/mjstevens777/9d6771c45f444843f9e3dce6a401b183)
    """

    def __init__(self, channels, kernel_size, stride, groups=1):
        """Set up the layer.
        Parameters
        ----------
        channels: int
            The number of input and output channels
        stride: int or tuple
            The amount of upsampling to do
        groups: int
            Set to 1 for a standard convolution. Set equal to channels to
            make sure there is no cross-talk between channels.
        """
        if isinstance(stride, int):
            stride = (stride, stride)

        assert groups in (1, channels), "Must use no grouping, " + \
                                        "or one group per channel"

        padding = (stride[0] - 1, stride[1] - 1)
        super().__init__(
            channels, channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups)

    def reset_parameters(self):
        """Reset the weight and bias."""
        init.constant(self.bias, 0)
        init.constant(self.weight, 0)
        bilinear_kernel = self.bilinear_kernel(self.kernel_size[0])
        for i in range(self.in_channels):
            if self.groups == 1:
                j = i
            else:
                j = 0
            self.weight.data[i, j] = bilinear_kernel

    @staticmethod
    def bilinear_kernel(kernel_size):
        """Generate a bilinear upsampling kernel."""
        bilinear_kernel = np.zeros([kernel_size, kernel_size])
        scale_factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = scale_factor - 1
        else:
            center = scale_factor - 0.5
        for x in range(kernel_size):
            for y in range(kernel_size):
                bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                                       (1 - abs(y - center) / scale_factor)

        return Tensor(bilinear_kernel)
