import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import os

PATH = "/home/julio/Off/AIDraw/procesamientoImagenes"

INPATH = PATH + "/rostros_listos"

OUTPATH = PATH + "/rostros_listos"

# CKPATH = PATH + ""

imgurls = os.listdir(INPATH) # ls -1 nombre de las imagenes entrada

n = 100
train_n = round(n*0.8)

# shuffle
randurls = np.copy(imgurls)
np.random.shuffle(randurls)

# train/test
tr_urls = randurls[:train_n]
ts_urls = randurls[train_n:n]

IMG_WIDTH = 256
IMG_HEIGHT = 256

def resize(inimg, tgimg, height, width):
    inimg = tf.image.resize(inimg, [height, width])
    tgimg = tf.image.resize(tgimg, [height, width])

    return inimg, tgimg

def normalize(inimg, tgimg):
    inimg = (inimg / 127.5) - 1
    tgimg = (tgimg / 127.5) - 1

    return inimg, tgimg

def random_jitter(inimg, tgimg):
    inimg, tgimg = resize(inimg, tgimg, 286, 286)

    stacked_image = tf.stack([inimg, tgimg], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    inimg, tgimg = cropped_image[0], cropped_image[1]

    if tf.random.uniform(()) > 0.5:
        inimg = tf.image.flip_left_right(inimg)
        tgimg = tf.image.flip_left_right(tgimg)

    return inimg, tgimg

def load_image(filename, augment=True):
    inimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(INPATH + '/' + filename)), tf.float32) [...,:3]
    reimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(INPATH + '/' + filename)), tf.float32) [...,:3]

    inimg, reimg = resize(inimg, reimg, IMG_HEIGHT, IMG_WIDTH)

    if augment:
        inimg, reimg = random_jitter(inimg, reimg)

    inimg, reimg = normalize(inimg, reimg)

    return inimg, reimg

def load_train_image(filename):
    return load_image(filename, True)

def load_test_image(filename):
    return load_image(filename, False)

img = ((load_train_image(randurls[0])[0]) + 1) / 2
# print('IMG', img)
import matplotlib
matplotlib.use('Qt5Agg')
# plt.imshow(img.numpy())
# plt.axis('off')
# plt.show()

train_dataset = tf.data.Dataset.from_tensor_slices(tr_urls)
train_dataset = train_dataset.map(load_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(1)

test_dataset = tf.data.Dataset.from_tensor_slices(ts_urls)
test_dataset = test_dataset.map(load_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = train_dataset.batch(1)

# -------------------------------generator model-------------------------------

from tensorflow.keras.layers import *
from tensorflow.keras import *

# blocks for the encoder
def downsample(filters, apply_batchnorm=True):
    result = Sequential()

    initializer = tf.random_normal_initializer(0, 0.02)

    # convolutional layer
    result.add(Conv2D(filters,
                      kernel_initializer=initializer,
                      strides=2,
                      padding="same",
                      kernel_size=4,
                      use_bias=not apply_batchnorm))

    # batchnorm layer
    if apply_batchnorm:
        result.add(BatchNormalization())

    # activation layer
    result.add(LeakyReLU())

    return result

# blocks for the decoder
def upsample(filters, appy_dropout=False):
    result = Sequential()

    initializer = tf.random_normal_initializer(0, 0.02)

    # convolutional layer
    result.add(Conv2DTranspose(filters,
                               kernel_initializer=initializer,
                               strides=2,
                               padding="same",
                               kernel_size=4,
                               use_bias=False))

    # batchnorm layer
    result.add(BatchNormalization())

    if appy_dropout:
        # dropout layer
        result.add(Dropout(0.5))

    # activation layer
    result.add(ReLU())

    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[None,None,3])

    down_stack = [
        downsample(64, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128),                       # (bs, 64, 64, 128)
        downsample(256),                       # (bs, 32, 32, 256)
        downsample(512),                       # (bs, 16, 16, 512)
        downsample(512),                       # (bs, 8, 8, 512)
        downsample(512),                       # (bs, 4, 4, 512)
        downsample(512),                       # (bs, 2, 2, 512)
        downsample(512),                       # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, appy_dropout=True),      # (bs, 2, 2, 1024)
        upsample(512, appy_dropout=True),      # (bs, 4, 4, 1024)
        upsample(512, appy_dropout=True),      # (bs, 8, 8, 1024)
        upsample(512),                         # (bs, 16, 16, 1024)
        upsample(256),                         # (bs, 32, 32, 512)
        upsample(128),                         # (bs, 64, 64, 256)
        upsample(64),                          # (bs, 128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0, 0.02)

    last = Conv2DTranspose(filters=3,
                           kernel_size=4,
                           strides=2,
                           padding="same",
                           kernel_initializer=initializer,
                           activation="tanh")
    # connections
    x = inputs
    s = []
    concat = Concatenate()



    for down in down_stack:
        x = down(x)
        s.append(x)

    s = reversed(s[:-1])

    for up, sk in zip(up_stack, s):
        x = up(x)
        x = concat([x, sk])

    last = last(x)

    return Model(inputs=inputs, outputs=last)

generator = Generator()

# -------------------------------discriminator model patchgang -------------------------
def Discriminator():
    ini = Input(shape=[None, None, 3], name="input_img")
    gen = Input(shape=[None, None, 3], name="gener_img")

    con = concatenate([ini, gen])

    initializer = tf.random_normal_initializer(0, 0.02)
    
    down1 = downsample(64, apply_batchnorm=False)(con)
    down2 = downsample(128)(down1)
    down3 = downsample(256)(down2)
    down4 = downsample(512)(down3)

    last = tf.keras.layers.Conv2D(filters=1,
                                  kernel_size=4,
                                  strides=1,
                                  kernel_initializer=initializer,
                                  padding="same")(down4)
    
    return tf.keras.Model(Inputs=[ini, gen], outputs=last)

discriminator = Discriminator

# -----------------------------const functions -------------------------------------

# calcula la entropia de las imagenes
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    # Diferencia entre los true po ser real y el detectado por el discriminador.
    # valor esperado todos 1 ya que es una imagen real
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    # Diferencia entre los false por ser generado y el detectado por el discriminador.
    # valor esperado todos 0 ya que es una imagen generada.
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
    # Diferencia entre lo que el discriminador nos da y lo que queremos que sean 1.
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # "Diferencia" media de lo generado y lo original esperamos que se aproximen mucho
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    # LAMBDA mas peso a l1_loss
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss

# -------------------------------------------------------------------------------
import os

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# checkpoint_prefix = os.path.join(CKPATH, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)

# checkpoint.restore(tf.train.latest_checkpoint(CKPATH)).asert_consumed()

def generate_images(model, test_input, tar, save_filename=False, display_imgs=True):
    prediction = model(test_input, training=True)

    if save_filename:
        tf.keras.preporcessing.image.save_img(PATH + '/output/' + save_filename + '.jpg', prediction[0, ...])
    
    plt.figure(figsize=(10,10))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    if display_imgs:
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')

    plt.show()

@tf.function()
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as discr_tape:

        output_image = generator(input_image, training=True)

        output_gen_discr = discriminator([output_image, input_image], training=True)
        output_trg_discr = discriminator([target, input_image], training=True)

        discr_loss = discriminator_loss(output_trg_discr, output_gen_discr)

        gen_loss = generator_loss(output_gen_discr, output_image, target)

        # back propagation
        generator_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        discriminator_grads = discr_tape.gradient(discr_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_grads, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator.trainable_variables))

from IPython.display import clear_output

def train(dataset, epochs):
    for epoch in range(epochs):
        imgi = 0
        for input_image, target in dataset:
            print('epoch' + str(epoch) + ' - train: ' + str(imgi)+'/'+str(len(tr_urls)))
            imgi+=1
            train_step(input_image, target)
            clear_output(wait=True)

        imgi = 0
        for inp, tar in test_dataset.take(5):
            generate_images(generator, inp, tar, str(imgi)+'_'+str(epoch), display_imgs=True)
            imgi += 1
        # if (epoch + 1) % 25 == 0:
        #     checkpoint.save(file_prefix=checkpoint_prefix)

# train(train_dataset, 100)