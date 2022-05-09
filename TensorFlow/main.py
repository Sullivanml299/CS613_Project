import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers

from IPython import display
from PIL import Image
import time
import pickle

from IPython import display

def imageToCSV():
    # sprite = 'front\\front_0000_0.png'
    # img = Image.open(sprite)
    # arr = np.asarray(img)
    # print(arr.shape)

    dir = 'front\\'
    for file in os.listdir(dir):
        sprite = os.path.join(dir, file)
        img = Image.open(sprite)
        arr = np.asarray(img)
        # print(arr.shape) #64x64x4
        with open("spriteData.csv", "ab") as f:
            if(arr.shape[2] == 4):
                arr = arr.reshape(1, -1)
                print(arr.shape)
                np.savetxt(f, arr, delimiter=",", newline='\n')
            f.close()


def arrayToImage(sprite):

    print(sprite.shape)
    sprite = sprite.numpy()
    spriteImage = Image.fromarray(sprite.astype(np.uint8))
    # spriteImage = Image.fromarray(sprite)
    spriteImage.show()


def generatePickleArr():
    arr = np.genfromtxt('spriteData.csv', delimiter=',', skip_header=False)
    arr = arr.reshape((arr.shape[0], 64, 64, 4))
    arr = arr[:,:,:, 0:3] #remove alpha layer
    print(arr.shape)
    file_pi = open('spriteArray.obj', 'wb')
    pickle.dump(arr, file_pi)

def restorePickleArr():
    file = open('spriteArray.obj', 'rb')
    arr = pickle.load(file)
    print(arr.shape)
    return arr


def tensorTest():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    print(train_images.shape)
    print(train_labels.shape)
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    print(train_images.shape)
    print(train_labels.shape)
    print(train_labels)
    # train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 3)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    print(model.output_shape)
    assert model.output_shape == (None, 64, 64, 3)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def spriteGAN():
    BUFFER_SIZE = 1256
    BATCH_SIZE = 256

    train_images = restorePickleArr()
    train_images = (train_images - 127.5) / 127.5  # FIXME: Normalize the images to [-1, 1]. May not want
    train_labels = np.ones((train_images.shape[0],))

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    print(train_dataset)

    generator = make_generator_model()

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    print(generated_image.shape)

    # arrayToImage(generated_image[0])
    # print(generated_image)
    # plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    # plt.show()

    discriminator = make_discriminator_model()
    decision = discriminator(generated_image)
    print(decision)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    #TRAINING LOOP
    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 1 # 16

    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    def generate_and_save_images(model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            # plt.subplot(4, 4, i + 1)
            # plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            # plt.axis('off')
            arrayToImage(predictions[0] * 127.5 + 127.5)

        # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        # plt.show()

    def train(dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                train_step(image_batch)

            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                                     epoch + 1,
                                     seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epochs,
                                 seed)

    train(train_dataset, EPOCHS)




# imageToCSV()
# csvToImage()
# generatePickleArr()
# restorePickleArr()


# tensorTest()
spriteGAN()
