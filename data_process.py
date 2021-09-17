import os
from PIL import Image
from PIL import UnidentifiedImageError
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from numpy.random import randn
from numpy.random import randint
from skimage.transform import resize
from numpy import asarray
from numpy import zeros
from numpy import ones
import cv2

def create_dir(save_dir, g_model):
    # devise name
    gen_shape = g_model.output_shape
    # model names contain current progression stage for easy resumption of training
    name = f'{gen_shape[1]}x{gen_shape[2]}'
    if not os.path.isdir(f'{save_dir}/'):
      os.mkdir(f'{save_dir}/')
    if not os.path.isdir(f'{save_dir}/{name}/'):
      os.mkdir(f'{save_dir}/{name}/')
    return name

def removeBrokenImg(DATA_DIR):
    for filename in os.listdir(DATA_DIR + '/1'):
        if filename.endswith(".jpg") or filename.endswith(".png"): 
            try:
                Image.open(DATA_DIR + '/1/' + filename)
            except UnidentifiedImageError:
                print('removed: ' + filename)
                os.remove(DATA_DIR + '/1/' + filename)

# preprocess function for images
# normalise between -1 to 1
def pre(X):
  X = X.astype('float32')
  X = (X - 127.5) / 127.5
  return X

# prepare image generator
real_gen = ImageDataGenerator(
        rescale=None,
        preprocessing_function=pre)

# generate a batch of real images
# real images are labelled 1
def generate_real_samples(real_generator):
  X_real, y_real = real_generator.next()
  return X_real, y_real + 1

# generate points in latent space as input for the generator
# these points are currently randomly generated
# TODO try using real images scaled down to 16x16 as input
def generate_latent_points(latent_dim, n_samples):
  # generate points in the latent space
  x_input = randn(latent_dim * n_samples)

  # reshape into a batch of inputs for the network
  x_input = x_input.reshape(n_samples, latent_dim)
  return x_input

# use the generator to generate n fake examples, with class labels
# fake images are labelled as -1
def generate_fake_samples(generator, latent_dim, n_samples):

	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
 
	# predict outputs
	X = generator.predict(x_input)
 
	# create class labels
	y = -ones((n_samples, 1))
 
	return X, y

# scale images to preferred size
def scale_dataset(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# creates a new dataset of the given resolution and saves in a specified folder
def scale_all_data(data_dir, new_shape, save_dir):
  out_dir = f'{save_dir}/resized_data/{new_shape[0]}x{new_shape[0]}/1/'
  if not os.path.isdir(f'{save_dir}/resized_data/'):
    os.mkdir(f'{save_dir}/resized_data/')
  if not os.path.isdir(f'{save_dir}/resized_data/{new_shape[0]}x{new_shape[0]}/'):
    os.mkdir(f'{save_dir}/resized_data/{new_shape[0]}x{new_shape[0]}/')
  if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

  for filename in os.listdir(data_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"): 
      print(f'--{filename} Done')
      im = cv2.imread(data_dir + filename)
      resized = cv2.resize(im, new_shape)
      cv2.imwrite(out_dir + '/' + filename, resized)

def prediction_post_process(X, file_name_head=None, batch_num=None):
    for i, x in enumerate(X):
      x = (x - x.min()) / (x.max() - x.min())
      x *= 255
      x = x.astype(np.uint8)
      X[i] = x
      if file_name_head:
          im = Image.fromarray(x)
          if not batch_num:
            im.save(file_name_head + f'_{str(i)}.png')
          else:
            im.save(file_name_head + f'_{str(i + batch_num)}.png')      

def predict_samples(g_model, latent_dim, save_dir, n_samples=10):
    name = create_dir(save_dir, g_model)
    # normalize pixel values to the range [0,1]
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    prediction_post_process(X, f'{save_dir}/{name}/plot')
    # for i in range(n_samples):
    #   x, _ = generate_fake_samples(g_model, latent_dim, 1)
    #   x = x[0]
    #   x = (x - x.min()) / (x.max() - x.min())
    #   x *= 255
    #   x = x.astype(np.uint8)
    #   # save plot to file
    # for i, x in enumerate(X):
    #     filename = f'{save_dir}/{name}/plot_{i}.png'
    #     im = Image.fromarray(x)
    #     im.save(filename)

    print(f'{n_samples} sample(s) generated at {save_dir}')



