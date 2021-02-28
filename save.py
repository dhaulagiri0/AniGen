from data_process import generate_fake_samples
from matplotlib import pyplot
from math import sqrt
from data_process import predict_samples
import tensorflow as tf
import pickle

# generate samples and save as a plot and save the model
def summarize_performance(status, wgan, latent_dim, n_blocks, cur_block, save_dir, n_samples=10):
    g_model = wgan.get_gen
    d_model = wgan.get_dis
    # devise name
    gen_shape = g_model.output_shape
    # model names contain current progression stage for easy resumption of training
    name = f'{gen_shape[1]}x{gen_shape[2]}-{status}-{n_blocks}-{cur_block}'
        # name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
    # generate images
            # X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
            # # normalize pixel values to the range [0,1]
            # X = (X - X.min()) / (X.max() - X.min())
            # # plot real images
            # square = int(sqrt(n_samples))
            # for i in range(n_samples):
            #     pyplot.subplot(square, square, 1 + i)
            #     pyplot.axis('off')
            #     pyplot.imshow(X[i])
            # # save plot to file
            # filename1 = save_dir + '/plot_%s.png' % (name)
            # pyplot.savefig(filename1)
            # pyplot.close()
    predict_samples(g_model, latent_dim, save_dir + f'/{name}/', n_samples)
    # save the generator model
    filename_g = save_dir + '/generator_%s.h5' % (name)
    filename_d = save_dir + '/discriminator_%s.h5' % (name)
    g_model.save(filename_g)
    d_model.save(filename_d)
    print('>Saved: %s, %s and %s' % ('_', filename_g, filename_d))

def generate_samples(status, cur_step, wgan, latent_dim, save_dir, n_samples=10):
    g_model = wgan.get_gen
    # devise name
    gen_shape = g_model.output_shape
    # model names contain current progression stage for easy resumption of training
    name = f'{gen_shape[1]}x{gen_shape[2]}-{status}-{cur_step}'
        # name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
    # generate images
    # X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # # normalize pixel values to the range [0,1]
    # X = (X - X.min()) / (X.max() - X.min())
    # # plot real images
    # square = int(sqrt(n_samples))
    # for i in range(n_samples):
    #     pyplot.subplot(square, square, 1 + i)
    #     pyplot.axis('off')
    #     pyplot.imshow(X[i])
    # # save plot to file
    # filename1 = save_dir + '/plot_%s.png' % (name)
    # pyplot.savefig(filename1)
    # pyplot.close()
    predict_samples(g_model, latent_dim, save_dir + f'/{name}/', n_samples)

def save_optimzers(wgan, save_dir, n_blocks, cur_block):
    gen_shape = wgan.get_gen.output_shape
    g_symbolic_weights = getattr(wgan.g_optimizer, 'weights')
    d_symbolic_weights = getattr(wgan.d_optimizer, 'weights')

    g_weight_values = tf.keras.backend.batch_get_value(g_symbolic_weights)
    d_weight_values = tf.keras.backend.batch_get_value(d_symbolic_weights)

    with open(f'{gen_shape}x{gen_shape}_{cur_block}_{n_blocks}_g_optimizer.pkl', 'wb') as f:
        pickle.dump(g_weight_values, f)
    with open(f'{gen_shape}x{gen_shape}_{cur_block}_{n_blocks}_d_optimizer.pkl', 'wb') as f:
        pickle.dump(d_weight_values, f)
