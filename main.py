from training import train, load_model
from discriminator import define_discriminator
from generator import define_generator
from keras.preprocessing.image import ImageDataGenerator
from data_process import pre


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Angio Dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train', 'resume' or 'inference'")
    parser.add_argument('--dataset_dir', required=True,
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
                        default= 6,
                        metavar="integer",
                        help='Number of conv blocks in decoder and encoder (supports up to 9 (1024x1024) of now)')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Data folder: ", args.dataset_dir)
    print("Save folder: ", args.save_dir)

    mode = args.command
    DATA_DIR = args.dataset_dir
    SAVE_DIR = args.save_dir
    
    # 4x, 8x, 16x, 32x, 64x, 128x, 256x, 512x, 1024x
    n_batch = [16, 16, 16, 16, 16, 16, 14, 6, 3]
    n_epochs = [5, 8, 8, 10, 10, 10, 12, 14, 16]

    if mode == 'train':
        # number of growth phases, e.g. 6 == [4, 8, 16, 32, 64, 128]
        n_blocks = args.n_blocks
        # size of the latent space
        latent_dim = 512

        # define base model
        d_base = define_discriminator()
        g_base = define_generator(latent_dim)

        # prepare image generator
        real_gen = ImageDataGenerator(
                rescale=None,
                preprocessing_function=pre)

        # train model
        train(g_base, d_base, latent_dim, n_epochs, n_epochs, n_batch, n_blocks, real_gen, DATA_DIR, SAVE_DIR)

    elif mode == 'resume':
        g_model_dir = args.g_model_path
        d_model_dir = args.d_model_path

        # size of the latent space
        latent_dim = 512

        wgan, n_blocks, cur_block = load_model(g_model_dir, d_model_dir, latent_dim)

        # prepare image generator
        real_gen = ImageDataGenerator(
                rescale=None,
                preprocessing_function=pre)

        # train model
        train(wgan.generator, wgan.discriminator, latent_dim, n_epochs, n_epochs, n_batch, n_blocks, real_gen, DATA_DIR, SAVE_DIR, cur_block)

    else:
        print('Not implemented')

