from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import initializers
from keras.models import load_model

import numpy as np
from numpy import zeros
from matplotlib import pyplot

random_dim = 100 #the dimension of random noise vector

def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(16384, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    return generator

def get_discriminator(optimizer):
    discriminator = Sequential()
    discriminator.add(Dense(1024, input_dim=16384, kernel_initializer=initializers.RandomNormal(stddev=0.2)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

    return discriminator

def get_gan(discriminator, random_dim, generator, optimizer):
    discriminator.trainable = False
    gan_input = Input(shape=(random_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    return gan

def train(epochs, batch_size, trainData):
    batch_count = trainData.shape[0]
    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan(discriminator, random_dim, generator, adam)

    for e in range(1, epochs+1):
        print('Epoch %d' % e)

        for i in range(batch_count):
            print('Batch %d' % i)
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = trainData[np.random.randint(0, trainData.shape[0], size=batch_size)]

            generated_images = generator.predict(noise)
            X = np.concatenate([image_batch, generated_images])

            y_dis = np.zeros(2*batch_size)
            y_dis[:batch_size] = 0.9

            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        #if e == 1 or e % 100 == 0:
        plot_generated_images(e, generator)

        if e == epochs:
            filename = 'model_gan.h5'
            generator.save(filename)
            print('Saved model')

def generate_image():
    noise = np.random.normal(0, 1, size=[100, random_dim])
    model = load_model('model_gan.h5')
    gen_image = model.predict(noise)
    print(gen_image)


def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 128, 128)

    pyplot.figure(figsize=figsize)

    for i in range(generated_images.shape[0]):
        pyplot.subplot(dim[0], dim[1], i+1)
        pyplot.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        pyplot.axis('off')

    pyplot.tight_layout
    pyplot.savefig('gan_generated_image_epoch_%d.png' % epoch)

