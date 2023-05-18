import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import time
from IPython import display
from tensorflow import keras
from keras import layers


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

def discriminador_loss(target, pred, gradient_penalty):
    c_lambda = 10
    d_loss = tf.math.reduce_mean(pred) - tf.math.reduce_mean(target) + c_lambda * gradient_penalty

    return d_loss

def generator_loss(fake):
    g_loss = -1 * tf.math.reduce_mean(fake)
    return g_loss


@tf.function
def grandient_Panalty(real, fake, epsilon):
    imgs = real * epsilon + fake * (1- epsilon)
    with tf.GradientTape() as tape:
        tape.watch(imgs)
        score = discriminador(imgs)
    gradient = tape.gradient(score, imgs)[0]
    gradient_norm = tf.norm(gradient)
    gp = tf.math.reduce_mean((gradient_norm-1)**2)
    return gp

(x_train, y_train),(_,_) = tf.keras.datasets.mnist.load_data()
#(x_train, y_train),(_,_) = tf.keras.datasets.fashion_mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], 28,28,1).astype('float32')
x_train = (x_train-127.5) / 127.5

buffersize = 60000
batchsize = 256
x_train = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffersize).batch(batchsize)

gerador = cria_gerador()
discriminador = cria_discriminador()

geradorOpt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
discriminadorOpt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

gerador.compile(optimizer= geradorOpt, loss = generator_loss)
discriminador.compile(optimizer= discriminadorOpt, loss = discriminador_loss)

checkpoint_dir = './training'
checkpoint_prefix = os.path.join(checkpoint_dir, "checkpoints")
checkpoint = tf.train.Checkpoint(geradorOpt, discriminadorOpt, gerador, discriminador)

epoch = 100
dim = 100
Num = 16

testImg = tf.random.normal([Num, dim])

@tf.function
def treinamento(imagens):
    ruido = tf.random.normal([batchsize, dim])
    dis_extra = 1
    for _ in range(dis_extra):
        with tf.GradientTape() as disc_tape:
            imgGerada = gerador(ruido, training = True)

            target = discriminador(imagens, training =True)
            pred = discriminador(imgGerada, training =True)

            epsilon = tf.random.normal([batchsize, 1, 1, 1], 0.0, 1.0)

            gp = grandient_Panalty(imagens, imgGerada, epsilon)

            d_loss = discriminador_loss(target, pred, gp)
        gradientes_disc = disc_tape.gradient(d_loss, discriminador.trainable_variables)
        discriminadorOpt.apply_gradients(zip(gradientes_disc, discriminador.trainable_variables))
    with tf.GradientTape() as gen_tape:
        imgGerada = gerador(ruido, training = True)
        pred = discriminador(imgGerada, training =True)

        g_loss = generator_loss(pred)

    gradientes_ger = gen_tape.gradient(g_loss, gerador.trainable_variables)
    geradorOpt.apply_gradients(zip(gradientes_ger, gerador.trainable_variables))

def train_gan(dataset, epoch, img_test):
    for i in range(epoch):
        inicio = time.time()
        for imgbatch in dataset:
            if(len(imgbatch) == batchsize):
                treinamento(imgbatch)
        display.clear_output(wait=True)
        gerar_salvar(gerador, i +1, testImg)
        if((i + 1) % (epoch / 10) == 0):
            checkpoint.save(file_prefix= checkpoint_prefix)
        print(i+1, time.time() - inicio)
    
    display.clear_output(wait= True)
    gerar_salvar(gerador, epoch, img_test)
    gerador.save('ger.h5')
 
def gerar_salvar(gerador, epoch, test):
    preds = gerador(test, training = False)
    fig = plt.figure(figsize=(4,4))
    for i in range(preds.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(preds[i, :, :, 0] * 127.5 + 127.5, cmap = 'gray')
        plt.axis('off')
    plt.savefig('img_epoch_{:04d}.png'.format(epoch))
    plt.show

train_gan(x_train, epoch, testImg)
