import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os

crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def cria_gerador():
    net = tf.keras.Sequential()

    net.add(layers.Dense(units = 7*7*256, use_bias = False, input_shape = (100,)))
    net.add(layers.BatchNormalization())
    net.add(layers.LeakyReLU())

    net.add(layers.Reshape((7,7,256)))

    net.add(layers.Conv2DTranspose(filters = 128, kernel_size = (5,5), padding = 'same', use_bias = False))
    net.add(layers.BatchNormalization())
    net.add(layers.LeakyReLU())

    net.add(layers.Conv2DTranspose(filters = 64, kernel_size = (5,5), padding = 'same', use_bias = False, strides = (2,2)))
    net.add(layers.BatchNormalization())
    net.add(layers.LeakyReLU())

    net.add(layers.Conv2DTranspose(filters = 1, kernel_size = (5,5), padding = 'same', use_bias = False, strides = (2,2), activation = 'tanh'))
    net.add(layers.BatchNormalization())
    net.add(layers.LeakyReLU())

    return net

def cria_discriminador():
    net = tf.keras.Sequential()

    net.add(layers.Conv2D(filters = 64, strides = (2,2), kernel_size = (5,5), padding = 'same', input_shape = [28,28,1]))
    net.add(layers.LeakyReLU())
    net.add(layers.Dropout(0.3))

    net.add(layers.Conv2D(filters = 128, strides=(2,2), kernel_size = (5,5), padding = 'same'))
    net.add(layers.LeakyReLU())
    net.add(layers.Dropout(0.3))

    net.add(layers.Flatten())
    net.add(layers.Dense(1))

    return net

def discriminador_loss(target, pred):
    real_loss = crossEntropy(tf.ones_like(target), target)
    fake_loss = crossEntropy(tf.zeros_like(pred), pred)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake):
    return crossEntropy(tf.ones_like(fake), fake)


#(x_train, y_train),(_,_) = tf.keras.datasets.mnist.load_data()
(x_train, y_train),(_,_) = tf.keras.datasets.fashion_mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], 28,28,1).astype('float')
x_train = (x_train-127.5) / 127.5

buffersize = 60000
batchsize = 256
x_train = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffersize).batch(batchsize)

gerador = cria_gerador()
discriminador = cria_discriminador()

geradorOpt = tf.keras.optimizers.Adam(learning_rate=0.00001)
discriminadorOpt = tf.keras.optimizers.Adam(learning_rate=0.00001)

gerador.compile(optimizer= geradorOpt, loss = generator_loss)
discriminador.compile(optimizer= discriminadorOpt, loss = discriminador_loss)

epoch = 100
dim = 100
Num = 16

testImg = tf.random.normal([Num, dim])
@tf.function
def treinamento(imagens):
    ruido = tf.random.normal([batchsize, dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        imgGerada = gerador(ruido, training = True)

        target = discriminador(imagens, training =True)
        pred = discriminador(imgGerada, training =True)

        gen_loss = generator_loss(pred)
        disc_loss = discriminador_loss(target, pred)

    gradientes_ger = gen_tape.gradient(gen_loss, gerador.trainable_variables)
    gradientes_disc = disc_tape.gradient(disc_loss, discriminador.trainable_variables)

    geradorOpt.apply_gradients(zip(gradientes_ger, gerador.trainable_variables))
    discriminadorOpt.apply_gradients(zip(gradientes_disc, discriminador.trainable_variables))

def train_gan(dataset, epoch, img_test):
    for i in range(epoch):
        for imgbatch in dataset:
            treinamento(imgbatch)
        print(i+1)
        if((i+1)%10 == 0):
            imgGer = gerador(img_test, training = False)
            fig = plt.figure(figsize=(10,10))
            for i in range(imgGer.shape[0]):
                plt.subplot(4,4,i+1)
                plt.imshow(imgGer[i,:,:,0] * 127.5 + 127.5, cmap='gray')
                plt.axis('off')
            plt.show()

train_gan(x_train, epoch, testImg)
