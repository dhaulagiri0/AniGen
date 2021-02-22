from math import sqrt
from matplotlib import pyplot
from data_process import generate_fake_samples, generate_real_samples, generate_latent_points
from wgan import WGAN
from generator import Generator
from discriminator import Discriminator
from custom_layers import WeightedSum, PixelNormalization, MinibatchStdev, Conv2DEQ, DenseEQ
import tensorflow.keras
from tensorflow.keras import backend, models
from losses import discriminator_loss, generator_loss
from save import summarize_performance, generate_samples
from tensorflow.keras.utils import plot_model

DATASET_SIZE = 63569

# update the alpha value on each instance of WeightedSum
# The alpha value of the weightedsum layer is gradually transitioned from 0 to 1 during the transition period when the model resolution is increased
def update_fadein(models, step, n_steps):
	# calculate current alpha (linear from 0 to 1)
	alpha = step / float(n_steps - 1)
	# update the alpha for each model
	for model in models:
		for layer in model.layers:
			if isinstance(layer, WeightedSum):
				backend.set_value(layer.alpha, alpha)

def load_model(g_dir, d_dir, latent_dim):
    g_name = g_dir.split('/')[-1]

    n_blocks = g_name.split('-')[-1].split(".")[-2]
    cur_block = g_name.split('-')[-2]

    cus = {
        'WeightedSum' : WeightedSum, 
        'PixelNormalization' : PixelNormalization,
        'MinibatchStdev' : MinibatchStdev,
        'Conv2DEQ' : Conv2DEQ,
        'DenseEQ' : DenseEQ
    }

    g_model = Generator(latent_dim)
    d_model = Discriminator()

    g_model.model = models.load_model(g_dir, custom_objects=cus, compile=False)
    d_model.model = models.load_model(d_dir, custom_objects=cus, compile=False)

    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=latent_dim,
        d_train = True,
        discriminator_extra_steps=1
    )

    return wgan, n_blocks, cur_block

# train a generator and discriminator
def train_epochs(wgan, real_generator, n_epochs, n_batch, save_dir, fadeIn=False):
    # calculate the number of batches per training epoch
    bat_per_epo = int(DATASET_SIZE / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    
    # define optimizers
    generator_optimizer = tensorflow.keras.optimizers.Adam(
        lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8
    )
    discriminator_optimizer = tensorflow.keras.optimizers.Adam(
        lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8
    )

    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )

    # manually enumerate epochs
    for i in range(n_steps):
        print(f'step {i + 1} out of {n_steps}')
        # update alpha for all WeightedSum layers when fading in new blocks
        if fadeIn:
            update_fadein([wgan.get_gen, wgan.get_dis], i, n_steps)
        # prepare real samples
        # the new wgan class no longer requires a y label
        X_real, _ = generate_real_samples(real_generator)
        # train the wgan for one batch
        losses = wgan.train_step(X_real)
        d_loss_real = float(losses['d_loss_real'])
        d_loss_fake = float(losses['d_loss_fake'])
        d_loss = float(losses['d_loss'])
        g_loss = float(losses['g_loss'])
        if (i+1) % bat_per_epo == 0:
            if fadeIn: status = 'fade'
            else: status = 'tune'
            generate_samples(status, i+1, wgan, wgan.latent_dim, save_dir)
        print(f'd_loss_real: {d_loss_real}  d_loss_fake: {d_loss_fake}  d_loss: {d_loss}  g_loss: {g_loss}')

# train the generator and discriminator
# real_gen is the keras image generator used to provide the real images
def train(wgan, latent_dim, e_norm, e_fadein, n_batch, n_blocks, real_gen, data_dir, save_dir, dynamic_resize, cur_block=0):
    # only runs this when we are training a model from scratch
    if cur_block == 0:
        # get the appropriate rescale size
        gen_shape = wgan.get_gen.output_shape
        # create new generator
        d = f'{data_dir}/resized_data/{gen_shape[1]}x{gen_shape[1]}/'
        if dynamic_resize: d = data_dir
        real_generator = real_gen.flow_from_directory(
                d,
                target_size=gen_shape[1:-1],
                batch_size=int(n_batch[0]),
                class_mode='binary')
        # train normal or straight-through models
        train_epochs(wgan, real_generator, e_norm[0], n_batch[0], save_dir)
        summarize_performance('tuned', wgan, latent_dim, 1, n_blocks, save_dir)
        cur_block += 1

    # process each level of growth
    for i in range(cur_block, n_blocks):
        print(i)
        # retrieve models for this level of growth
        wgan.generator.add_generator_block(i)
        wgan.discriminator.add_discriminator_block(i)

        # get the appropriate rescale size
        gen_shape = wgan.get_gen.output_shape
        # create new generator
        d = f'{data_dir}/resized_data/{gen_shape[1]}x{gen_shape[1]}/'
        if dynamic_resize: d = data_dir
        real_generator = real_gen.flow_from_directory(
                d,
                target_size=gen_shape[1:-1],
                batch_size=int(n_batch[i]),
                class_mode='binary')
        # train fade-in models for next level of growth
        train_epochs(wgan, real_generator, e_fadein[i], n_batch[i], save_dir, True)
        summarize_performance('faded', wgan, latent_dim, i+1, n_blocks, save_dir)
        # switch to normal model and tune
        wgan.generator.switch()
        wgan.discriminator.switch()
        train_epochs(wgan, real_generator, e_norm[i], n_batch[i], save_dir)
        summarize_performance('tuned', wgan, latent_dim, i+1, n_blocks, save_dir)
