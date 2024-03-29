import os
import numpy as np
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
from training import train, load_model, load_generator, extra_epochs
from discriminator import Discriminator
from generator import Generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_process import pre, scale_all_data
from wgan import WGAN
from data_process import predict_samples
from traversal import traverse_latent_space, genGif, read_latent_points

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Angio Dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train', 'resume', 'predict', 'traverse")
    parser.add_argument('--dataset_dir', required=False,
                        metavar="/path/to/angio/",
                        help="Directory of the dataset")
    parser.add_argument('--g_model_path', required=False,
                        metavar="/path/to/g_model_weights.h5",
                        help="Path to weights .h5 file of the generator")
    parser.add_argument('--d_model_path', required=False,
                        metavar="/path/to/d_model_weights.h5",
                        help="Path to weights .h5 file of the discriminator")
    parser.add_argument('--save_dir', required=True,
                        metavar="/path/to/logs/",
                        help='Model checkpoints and sample plots directory')
    parser.add_argument('--n_blocks', required=False,
                        default= 9,
                        metavar="integer",
                        help='Number of conv blocks in decoder and encoder (supports up to 9 (1024x1024) of now)')
    parser.add_argument('--save_resized_data', required=False,
                        dest='save_resize',
                        action='store_true',
                        help='Whether to save the resized dataset for future use')
    parser.add_argument('--dynamic_resize', required=False,
                        dest='dynamic_resize',
                        action='store_true',
                        help='Whether to keep the resized data in memory')
    parser.add_argument('--n_samples', required=False,
                        dest='n_samples',
                        help='Number of samples to predict')
    parser.add_argument('--extra_epochs', required=False,
                        dest='extra_epochs',
                        action='store_true',
                        help='Number of extra tune epochs to train')
    parser.add_argument('--gen_gif', required=False,
                        dest='gen_gif',
                        action='store_true',
                        default=False,
                        help='whether to generate gif after latent space traversal')
    parser.add_argument('--dims_path', required=False,
                        dest='dims_path',
                        help='path to nptxt file storing latent dims')
    parser.set_defaults(save_resize = False, dynamic_resize=False, extra_epochs=False)

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Data folder: ", args.dataset_dir)
    print("Save folder: ", args.save_dir)

    mode = args.command
    DATA_DIR = args.dataset_dir
    SAVE_DIR = args.save_dir
    save_resize = args.save_resize
    # number of growth phases, e.g. 6 == [4, 8, 16, 32, 64, 128]
    n_blocks = int(args.n_blocks)
    
    # 4x, 8x, 16x, 32x, 64x, 128x, 256x, 512x, 1024x
    n_batch = [128, 128, 64, 32, 6, 4, 2, 1, 1]
    n_epochs = [8, 10, 10, 10, 10, 15, 15, 15, 20]

    if mode == 'train':
        # size of the latent space
        latent_dim = 512

        # define base model
        d_base = Discriminator()
        g_base = Generator(latent_dim)
        wgan = WGAN(
                discriminator=d_base,
                generator=g_base,
                latent_dim=latent_dim,
                d_train = True,
                discriminator_extra_steps=1
        )

        # prepare image generator
        real_gen = ImageDataGenerator(
                rescale=None,
                preprocessing_function=pre)

        # train model
        train(wgan, latent_dim, n_epochs, n_epochs, n_batch, n_blocks, real_gen, DATA_DIR, SAVE_DIR, args.dynamic_resize)

    elif mode == 'resume':
        g_model_dir = args.g_model_path
        d_model_dir = args.d_model_path

        # size of the latent space
        latent_dim = 512

        wgan, n_blocks, cur_block = load_model(g_model_dir, d_model_dir, latent_dim)
        print('loaded')

        # prepare image generator
        real_gen = ImageDataGenerator(
                rescale=None,
                preprocessing_function=pre)

        # train model
        if not args.extra_epochs:
            train(wgan, latent_dim, n_epochs, n_epochs, n_batch, int(n_blocks), real_gen, DATA_DIR, SAVE_DIR, args.dynamic_resize, int(cur_block))
        else:
            extra_epochs(wgan, latent_dim, n_epochs, n_epochs, n_batch, int(n_blocks), real_gen, DATA_DIR, SAVE_DIR, args.dynamic_resize, int(cur_block))
            train(wgan, latent_dim, n_epochs, n_epochs, n_batch, int(n_blocks), real_gen, DATA_DIR, SAVE_DIR, args.dynamic_resize, int(cur_block))
    elif mode =='predict':
        g_model_dir = args.g_model_path

        # size of the latent space
        latent_dim = 512

        g_model = load_generator(g_model_dir, latent_dim)

        n_samples = int(args.n_samples)
        save_dir = args.save_dir
        predict_samples(g_model.model, latent_dim, save_dir, n_samples)
    elif mode == 'traverse':
        g_model_dir = args.g_model_path

        latent_dim = 512

        g_model = load_generator(g_model_dir, latent_dim)

        n_samples = int(args.n_samples)
        save_dir = args.save_dir

        if args.dims_path:
            dims_path = args.dims_path
            latent_points = read_latent_points(dims_path, latent_dim, n_samples)
        else:
            latent_points = None
        # generate images 
        file_names, latent_dims = traverse_latent_space(g_model.model, latent_dim, save_dir, n_samples=n_samples, latent_points=latent_points)
        if not args.dims_path:
            with open(f'{save_dir}/dims_{latent_dim}_{len(latent_dims)}.txt', 'w') as f:
                for row in latent_dims:
                    np.savetxt(f, row)
        print('Saved all latent dims')

        gen_shape = g_model.model.output_shape
        name = f'{gen_shape[1]}x{gen_shape[2]}'
        # generate gif
        if args.gen_gif:
            genGif(save_dir + f'/{name}/', save_dir)
            print('Gif generation successful')
    else:
        print('Not implemented')
