from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, AveragePooling2D, LeakyReLU, Layer, Add
from keras.constraints import max_norm
from keras.initializers import RandomNormal
from custom_layers import WeightedSum, MinibatchStdev, Conv2DEQ


# add a discriminator block
def add_discriminator_block(old_model, cur_block, n_input_layers=3):
  filters = [512, 512, 512, 256, 128, 64, 32, 16]
  f = filters[cur_block - 1]
  # weight initialization
  init = RandomNormal(mean=0., stddev=1.)
  # get shape of existing model
  in_shape = list(old_model.input.shape)
  # define new input shape as double the size
  input_shape = (in_shape[-2]*2, in_shape[-2]*2, in_shape[-1])
  in_image = Input(shape=input_shape)
  # define new input processing layer
  if cur_block > 3:
    d = Conv2DEQ(f/2, (1,1), padding='same', kernel_initializer=init)(in_image)
  else:
    d = Conv2DEQ(f, (1,1), padding='same', kernel_initializer=init)(in_image)
  d = LeakyReLU(alpha=0.2)(d)
  # define new block
  if cur_block > 3:
    d = Conv2DEQ(f/2, (3,3), padding='same', kernel_initializer=init)(d)
  else:
    d = Conv2DEQ(f, (3,3), padding='same', kernel_initializer=init)(d)
  d = LeakyReLU(alpha=0.2)(d)
  d = Conv2DEQ(f, (3,3), padding='same', kernel_initializer=init)(d)
  d = LeakyReLU(alpha=0.2)(d)
  d = AveragePooling2D()(d)
  block_new = d
  # skip the input, 1x1 and activation for the old model
  for i in range(n_input_layers, len(old_model.layers)):
    d = old_model.layers[i](d)
  # define straight-through model
  model1 = Model(in_image, d)
  # compile model
  # model1.compile(loss='mse', optimizer=discriminator_optimizer)
  # downsample the new larger image
  downsample = AveragePooling2D()(in_image)
  # connect old input processing to downsampled new input
  block_old = old_model.layers[1](downsample)
  block_old = old_model.layers[2](block_old)
  # fade in output of old model input layer with new input
  d = WeightedSum()([block_old, block_new])
  # skip the input, 1x1 and activation for the old model
  for i in range(n_input_layers, len(old_model.layers)):
    d = old_model.layers[i](d)
  # define straight-through model
  model2 = Model(in_image, d)
  # compile model
  # model2.compile(loss=wasserstein_loss, optimizer=discriminator_optimizer)
  return [model1, model2]

 
# define the discriminator models for each image resolution
def define_discriminator(input_shape=(4,4,3)):
  # weight initialization
  init = RandomNormal(mean=0., stddev=1.)
  # base model input
  in_image = Input(shape=input_shape)
  # conv 1x1
  d = Conv2DEQ(512, (1,1), padding='same', kernel_initializer=init)(in_image)
  d = LeakyReLU(alpha=0.2)(d)
  # conv 3x3 (output block)
  d = MinibatchStdev()(d)
  d = Conv2DEQ(512, (3,3), padding='same', kernel_initializer=init)(d)
  d = LeakyReLU(alpha=0.2)(d)
  # conv 4x4
  d = Conv2DEQ(512, (4,4), padding='same', kernel_initializer=init)(d)
  d = LeakyReLU(alpha=0.2)(d)
  # dense output layer
  d = Flatten()(d)
  out_class = Dense(1, activation='linear')(d)
  # define model
  model = Model(in_image, out_class)
  # compile model
  # model.compile(loss=wasserstein_loss, optimizer=discriminator_optimizer)
      # LEGACY
      # # store model
      # model_list.append([model, model])
      # # create submodels
      # for i in range(1, n_blocks):
      #   # get prior model without the fade-on
      #   old_model = model_list[i - 1][0]
      #   # create new model for next resolution
      #   models = add_discriminator_block(old_model)
      #   # store model
      #   model_list.append(models)
      # return model_list
  return model
