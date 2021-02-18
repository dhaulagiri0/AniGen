from math import sqrt
from matplotlib import pyplot
from data_process import generate_fake_samples, generate_real_samples, generate_latent_points
from wgan import WGAN
from generator import add_generator_block
from discriminator import add_discriminator_block
from custom_layers import WeightedSum, PixelNormalization, MinibatchStdev, Conv2DEQ
import keras
from keras import backend, models
from losses import discriminator_loss, generator_loss
from save import summarize_performance

DATASET_SIZE = 320

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
        'Conv2DEQ' : Conv2DEQ
    }

    g_model = models.load_model(g_dir, custom_objects=cus, compile=False)
    d_model = models.load_model(d_dir, custom_objects=cus, compile=False)

    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=latent_dim,
        d_train = True,
        discriminator_extra_steps=5
    )

    return wgan, n_blocks, cur_block

# train a generator and discriminator
def train_epochs(wgan, real_generator, n_epochs, n_batch, fadein=False):
    # calculate the number of batches per training epoch
    bat_per_epo = int(DATASET_SIZE / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    # half_batch = int(n_batch / 2)

    # define optimizers
    generator_optimizer = keras.optimizers.Adam(
        lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8
    )
    discriminator_optimizer = keras.optimizers.Adam(
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
        if fadein:
            update_fadein([wgan], i, n_steps)
        # prepare real samples
        # the new wgan class no longer requires a y label
        X_real, _ = generate_real_samples(real_generator)
        # train the wgan for one batch
        losses = wgan.train_step(X_real)
        d_loss_real = float(losses['d_loss_real'])
        d_loss_fake = float(losses['d_loss_fake'])
        d_loss = float(losses['d_loss'])
        g_loss = float(losses['g_loss'])
        print(f'd_loss_real: {d_loss_real}  d_loss_fake: {d_loss_fake}  d_loss: {d_loss}  g_loss: {g_loss}')
        
        # wgan.train_on_batch(X_real)

            # LEGACY
            # X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # # print('Fake input shape: ', X_fake.shape)    
            # # update discriminator model
            # d_loss1 = d_model.train_on_batch(X_real, y_real)
            # d_loss2 = d_model.train_on_batch(X_fake, y_fake)
            # # update the generator via the discriminator's error
            # z_input = generate_latent_points(latent_dim, n_batch)
            # y_real2 = ones((n_batch, 1))
            # g_loss = gan_model.train_on_batch(z_input, y_real2)
            # # summarize loss on this batch
            # print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, d_loss1, d_loss2, g_loss))

# train the generator and discriminator
# real_gen is the keras image generator used to provide the real images
def train(g_model, d_model, latent_dim, e_norm, e_fadein, n_batch, n_blocks, real_gen, data_dir, save_dir, cur_block=0):
    # only runs this when we are training a model from scratch
    if cur_block == 0:
        # fit the baseline model
        g_normal, d_normal = g_model, d_model
        wgan = WGAN(
                discriminator=d_normal,
                generator=g_normal,
                latent_dim=latent_dim,
                d_train = True,
                discriminator_extra_steps=1
        )
        # get the appropriate rescale size
        gen_shape = g_normal.output_shape
        # create new generator
        real_generator = real_gen.flow_from_directory(
                data_dir,
                target_size=gen_shape[1:-1],
                batch_size=int(n_batch[0]),
                class_mode='binary')
        # train normal or straight-through models
        train_epochs(wgan, real_generator, e_norm[0], n_batch[0])
        summarize_performance('tuned', wgan, latent_dim, 1, n_blocks, save_dir)
        cur_block += 1
    else:
        wgan = WGAN(
		discriminator=d_model,
		generator=g_model,
		latent_dim=latent_dim,
		d_train = True,
		discriminator_extra_steps=1
	)

    # process each level of growth
    for i in range(cur_block, n_blocks):
        print(i)
        # retrieve models for this level of growth
        [g_normal, g_fadein] = add_generator_block(wgan.generator, i)
        [d_normal, d_fadein] = add_discriminator_block(wgan.discriminator, i)
        # [gan_normal, gan_fadein] = gan_models[i]
        # update the existing wgan to fade in stage
        wgan.generator = g_fadein
        wgan.discriminator = d_fadein
            # LEGACY
            # wgan_fade = WGAN(
            #     discriminator=d_fadein,
            #     generator=g_fadein,
            #     latent_dim=latent_dim,
            #     d_train = True,
            #     discriminator_extra_steps=5
            # )
        # get the appropriate rescale size
        gen_shape = g_normal.output_shape
        # create new generator
        real_generator = real_gen.flow_from_directory(
                data_dir,
                target_size=gen_shape[1:-1],
                batch_size=int(n_batch[i]),
                class_mode='binary')
        # train fade-in models for next level of growth
        train_epochs(wgan, real_generator, e_fadein[i], n_batch[i], True)
        summarize_performance('faded', wgan, latent_dim, i+1, n_blocks, save_dir)
        # update wgan to normal mode and train
        wgan.generator = g_normal
        wgan.discriminator = d_normal
            # LEGACY
            # wgan_normal = WGAN(
            #     discriminator=d_normal,
            #     generator=g_normal,
            #     latent_dim=latent_dim,
            #     d_train = False,
            #     discriminator_extra_steps=5
            # )
        train_epochs(wgan, real_generator, e_norm[i], n_batch[i])
        summarize_performance('tuned', wgan, latent_dim, i+1, n_blocks, save_dir)
