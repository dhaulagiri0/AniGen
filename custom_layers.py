import tensorflow as tf
import tensorflow.keras
import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.layers import Layer, Add, Conv2D, Dense

# pixel-wise feature vector normalization layer
class PixelNormalization(Layer):
	def __init__(self, **kwargs):
		super(PixelNormalization, self).__init__(**kwargs)

	def call(self, inputs):
		# calculate square pixel values
		values = inputs**2.0
		# calculate mean pixel values
		mean_values = backend.mean(values, axis=-1, keepdims=True)
		# ensure mean is not zero
		mean_values += 1.0e-8
		# L2 norm
		l2 = backend.sqrt(mean_values)
		# normalize values by l2 norm
		normalized = inputs / l2
		return normalized

	def compute_output_shape(self, input_shape):
		return input_shape

class MinibatchStdev(Layer):
	def __init__(self, **kwargs):
		super(MinibatchStdev, self).__init__(**kwargs)
 
	def call(self, inputs):
		# calculate mean value for each pixel across channels
		mean = backend.mean(inputs, axis=0, keepdims=True)
		# calculate squared differences
		squ_diffs = backend.square(inputs - mean)
		# variance
		mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims=True)
		mean_sq_diff += 1e-8
		# stdev
		stdev = backend.sqrt(mean_sq_diff)
		# calculate mean standard deviation across each pixel
		mean_pix = backend.mean(stdev, keepdims=True)

		shape = backend.shape(inputs)
		output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
		# concatenate with the output
		combined = backend.concatenate([inputs, output], axis=-1)
		return combined
 
	def compute_output_shape(self, input_shape):
		input_shape = list(input_shape)
		# add one to the channel dimension (assume channels-last)
		input_shape[-1] += 1
		return tuple(input_shape)   

class WeightedSum(Add):
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')

    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output 

    def get_config(self):
        config = super().get_config().copy()
        config.update({"alph":self.alpha.numpy(),})
        return config

class Conv2DEQ(Conv2D):
    def __init__(self, *args, **kwargs):
        super(Conv2DEQ, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        super(Conv2DEQ, self).build(input_shape)
        # The number of inputs
        n = np.product([int(val) for val in input_shape[1:]])
        # He initialisation constant
        self.c = np.sqrt(2/n)

    def call(self, inputs):
        if self.rank == 2:
            outputs = backend.conv2d(
                inputs,
                self.kernel*self.c, # scale kernel
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = backend.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

class DenseEQ(Dense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        # The number of inputs
        n = np.product([int(val) for val in input_shape[1:]])
        # He initialisation constant
        self.c = np.sqrt(2/n)

    def call(self, inputs):
        output = backend.dot(inputs, self.kernel*self.c) # scale kernel
        if self.use_bias:
            output = backend.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output