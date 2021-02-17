import os
from PIL import Image
from PIL import UnidentifiedImageError
from keras.preprocessing.image import ImageDataGenerator
from numpy.random import randn
from numpy.random import randint
from skimage.transform import resize
from numpy import asarray
from numpy import zeros
from numpy import ones

def removeBrokenImg(DATA_DIR):
    for filename in os.listdir(DATA_DIR + '/1'):
        if filename.endswith(".jpg"): 
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