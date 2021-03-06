import tensorflow as tf
import tensorflow.keras

class WGAN(tensorflow.keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        d_train,
        discriminator_extra_steps=1,
        gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.d_train = d_train

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile() # this line for some reason causes problem is tensorflow version is not new enough
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.get_dis(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        if self.d_train:
            for i in range(self.d_steps):
                # Get the latent vector
                random_latent_vectors = tf.random.normal(
                    shape=(batch_size, self.latent_dim)
                )
                with tf.GradientTape() as tape:
                    # Generate fake images from the latent vector
                    fake_images = self.get_gen(random_latent_vectors, training=True)
                    # Get the logits for the fake images
                    fake_logits = self.get_dis(fake_images, training=True)
                    # Get the logits for the real images
                    real_logits = self.get_dis(real_images, training=True)

                    # Calculate the discriminator loss using the fake and real image logits
                    d_loss_real, d_loss_fake, d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                    # Calculate the gradient penalty
                    gp = self.gradient_penalty(batch_size, real_images, fake_images)
                    # Add the gradient penalty to the original discriminator loss
                    d_loss = d_cost + gp * self.gp_weight
                # Get the gradients w.r.t the discriminator loss
                d_gradient = tape.gradient(d_loss, self.get_dis.trainable_variables)
                # Update the weights of the discriminator using the discriminator optimizer
                self.d_optimizer.apply_gradients(
                    zip(d_gradient, self.get_dis.trainable_variables)
                )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.get_gen(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.get_dis(generated_images, training=False)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)
            
        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.get_gen.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.get_gen.trainable_variables)
        )
                # LEGACY PRINTING
                # print(f'd_loss: {float(d_loss)}  g_loss: {float(g_loss)}')
        return {"d_loss_real": d_loss_real, "d_loss_fake": d_loss_fake,"d_loss": d_loss, "g_loss": g_loss}

    @property
    def get_gen(self):
        return self.generator.model

    @property
    def get_dis(self):
        return self.discriminator.model
