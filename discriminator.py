import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, AveragePooling2D, LeakyReLU, Layer, Add
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.initializers import RandomNormal
from custom_layers import WeightedSum, MinibatchStdev, Conv2DEQ, DenseEQ

class Discriminator:
    
  def __init__(self):
      with tf.name_scope('discriminator'):
          self.define_discriminator()


  # add a discriminator block
  def add_discriminator_block(self, cur_block, n_input_layers=3):
    old_model = self.model
    filters = [512, 512, 512, 512, 256, 128, 64, 32]
    f = filters[cur_block - 1]
    # get shape of existing model
    in_shape = list(old_model.input.shape)
    # define new input shape as double the size
    input_shape = (in_shape[-2]*2, in_shape[-2]*2, in_shape[-1])
    in_image = Input(shape=input_shape)
    # define new input processing layer
    if cur_block > 3:
      d = Conv2DEQ(int(f/2), (1,1), padding='same', name='d_conv_' + str(cur_block) + '_1')(in_image)
    else:
      d = Conv2DEQ(f, (1,1), padding='same', name='d_conv_' + str(cur_block) + '_1')(in_image)
    d = LeakyReLU(alpha=0.2, name='d_relu_' + str(cur_block) + '_1')(d)
    # define new block
    if cur_block > 3:
      d = Conv2DEQ(int(f/2), (3,3), padding='same', name='d_conv_' + str(cur_block) + '_2')(d)
    else:
      d = Conv2DEQ(f, (3,3), padding='same', name='d_conv_' + str(cur_block) + '_2')(d)
    d = LeakyReLU(alpha=0.2, name='d_relu_' + str(cur_block) + '_2')(d)
    d = Conv2DEQ(f, (3,3), padding='same', name='d_conv_' + str(cur_block) + '_3')(d)
    d = LeakyReLU(alpha=0.2, name='d_relu_' + str(cur_block) + '_3')(d)
    d = AveragePooling2D(name='d_avgpool_' + str(cur_block) + '_1')(d)
    block_new = d
    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(old_model.layers)):
      d = old_model.layers[i](d)
    # define straight-through model
    model1 = Model(in_image, d)
    # downsample the new larger image
    downsample = AveragePooling2D(name='d_avgpool_' + str(cur_block) + '_2')(in_image)
    # connect old input processing to downsampled new input
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)
    # fade in output of old model input layer with new input
    d = WeightedSum(name='d_wsum_' + str(cur_block) + '_1')([block_old, block_new])
    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(old_model.layers)):
      d = old_model.layers[i](d)
    # define fade-in model
    model2 = Model(in_image, d)
    self.normal = model1
    # set cur model to fade in
    self.model = model2
    # return [model1, model2]

  
  # define base discriminator
  def define_discriminator(self, input_shape=(4,4,3)):
    # base model input
    in_image = Input(shape=input_shape)
    # conv 1x1
    d = Conv2DEQ(512, (1,1), padding='same', name='d_conv_0_1')(in_image)
    d = LeakyReLU(alpha=0.2, name='d_relu_0_1')(d)
    # conv 3x3 (output block)
    d = MinibatchStdev(name='d_ministdev_0_1')(d)
    d = Conv2DEQ(512, (3,3), padding='same', name='d_conv_0_2')(d)
    d = LeakyReLU(alpha=0.2, name='d_relu_0_2')(d)
    # conv 4x4
    d = Conv2DEQ(512, (4,4), padding='valid', name='d_conv_0_3')(d)
    d = LeakyReLU(alpha=0.2, name='d_relu_0_3')(d)
    # dense output layer
    d = Flatten(name='d_flatten_0_1')(d)
    out_class = DenseEQ(1, name='d_dense_0_1')(d)
    # define model
    model = Model(in_image, out_class)
    self.model = model

  def switch(self):
    # replace fade in model with normal model
    self.model = self.normal
