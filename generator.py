from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, AveragePooling2D, LeakyReLU, Layer, Add
from keras.constraints import max_norm
from keras.initializers import RandomNormal
from custom_layers import WeightedSum, MinibatchStdev, PixelNormalization, Conv2DEQ

# add a generator block
def add_generator_block(old_model, cur_block):
  filters = [512, 512, 512, 256, 128, 64, 32, 16]
  f = filters[cur_block - 1]
  # weight initialization
  init = RandomNormal(mean=0., stddev=1.)
  # get the end of the last block
  block_end = old_model.layers[-2].output
  # upsample, and define new block
  upsampling = UpSampling2D(name='g_up2d_' + str(cur_block))(block_end)
  g = Conv2DEQ(f, (3,3), padding='same', kernel_initializer=init, name='g_conv_' + str(cur_block) + '_1')(upsampling)
  g = PixelNormalization(name='g_pxnorm_' + str(cur_block) + '_1')(g)
  g = LeakyReLU(alpha=0.2, name='g_relu_' + str(cur_block) + '_1')(g)
  g = Conv2DEQ(f, (3,3), padding='same', kernel_initializer=init, name='g_conv_' + str(cur_block) + '_2')(g)
  g = PixelNormalization(name='g_pxnorm_' + str(cur_block) + '_2')(g)
  g = LeakyReLU(alpha=0.2, name='g_relu_' + str(cur_block) + '_2')(g)
  # add new output layer
  out_image = Conv2DEQ(3, (1,1), padding='same', kernel_initializer=init, name='g_conv_' + str(cur_block) + '_3')(g)
  # define model
  model1 = Model(old_model.input, out_image)
  # get the output layer from old model
  out_old = old_model.layers[-1]
  # connect the upsampling to the old output layer
  out_image2 = out_old(upsampling)
  # define new output image as the weighted sum of the old and new models
  merged = WeightedSum(name='g_wsum_' + str(cur_block) + '_1')([out_image2, out_image])
  # define model
  model2 = Model(old_model.input, merged)
  return [model1, model2]
 
 
# define generator models
def define_generator(latent_dim, in_dim=4):
  # weight initialization
  init = RandomNormal(mean=0., stddev=1.)
  # base model latent input
  in_latent = Input(shape=(latent_dim,))
  # linear scale up to activation maps
  g  = Dense(512 * in_dim * in_dim, kernel_initializer=init, name='g_dense_0_1')(in_latent)
  g = Reshape((in_dim, in_dim, 512), name='g_reshape_0_1')(g)
  # conv 4x4, input block
  g = Conv2DEQ(512, (3,3), padding='same', kernel_initializer=init, name='g_conv_0_1')(g)
  g = PixelNormalization(name='g_pxnorm_0_1')(g)
  g = LeakyReLU(alpha=0.2, name='g_relu_0_1')(g)
  # conv 3x3
  g = Conv2DEQ(512, (3,3), padding='same', kernel_initializer=init, name='g_conv_0_2')(g)
  g = PixelNormalization(name='g_pxnorm_0_2')(g)
  g = LeakyReLU(alpha=0.2, name='g_relu_0_2')(g)
  # conv 1x1, output block
  out_image = Conv2DEQ(3, (1,1), padding='same', kernel_initializer=init, name='g_conv_0_3')(g)
  # define model
  model = Model(in_latent, out_image)
      # LEGACY
      # store model
      # model_list.append([model, model])
      # create submodels
      # for i in range(1, n_blocks):
      # 	# get prior model without the fade-on
      # 	old_model = model_list[i - 1][0]
      # 	# create new model for next resolution
      # 	models = add_generator_block(old_model)
      # 	# store model
      # 	model_list.append(models)
      # return model_list
  return model
