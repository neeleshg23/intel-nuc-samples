from intel_npu_acceleration_library.backend import Convolution
import numpy as np
input_shape = [32, 32, 3]
weights_shape = [64, 3, 3, 3]

conv_layer = Convolution(
    input_shape=input_shape,
    weights_shape=weights_shape,
    bias=True,
    strides=1,
    padding=1,
    dilation=1,
    groups=1,
    profile=True,
    device="NPU"
)

result = conv_layer.run()
