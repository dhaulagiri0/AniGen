from data_process import scale_all_data

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Angio Dataset.')
    parser.add_argument('--dataset_dir', required=True,
                        metavar="/path/to/angio/",
                        help="Directory of the dataset")
    parser.add_argument('--n_blocks', required=False,
                        default= 9,
                        metavar="integer",
                        help='Number of conv blocks in decoder and encoder (supports up to 9 (1024x1024) of now)')

    args = parser.parse_args()
    print("Data folder: ", args.dataset_dir)

    DATA_DIR = args.dataset_dir
    # number of growth phases, e.g. 6 == [4, 8, 16, 32, 64, 128]
    n_blocks = int(args.n_blocks)

    # create data for each dimension
    for i in range(0, n_blocks):
        dim = 4 * 2**i
        scale_all_data(DATA_DIR, (dim, dim))
