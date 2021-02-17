# generate samples and save as a plot and save the model
def summarize_performance(status, wgan, latent_dim, n_blocks, cur_block, n_samples=25):
    g_model = wgan.generator
    d_model = wgan.discriminator
    # devise name
    gen_shape = g_model.output_shape
    # model names contain current progression stage for easy resumption of training
    name = f'{gen_shape[1]}x{gen_shape[2]}-{status}-{n_blocks}-{cur_block}'
        # name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
    # generate images
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # normalize pixel values to the range [0,1]
    X = (X - X.min()) / (X.max() - X.min())
    # plot real images
    square = int(sqrt(n_samples))
    for i in range(n_samples):
        pyplot.subplot(square, square, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X[i])
    # save plot to file
    filename1 = DATA_DIR + '/plot_%s.png' % (name)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename_g = DATA_DIR + '/generator_%s.h5' % (name)
    filename_d = DATA_DIR + '/discriminator_%s.h5' % (name)
    g_model.save(filename_g)
    d_model.save(filename_d)
    print('>Saved: %s, %s and %s' % (filename1, filename_g, filename_d))