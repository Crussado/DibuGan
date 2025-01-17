import tensorflow as tf
import os

def downsample(filters, size, apply_batchnorm=True):
    result = tf.keras.Sequential()

    initializer = tf.random_normal_initializer(0, 0.02)

    # convolutional layer
    result.add(tf.keras.layers.Conv2D(filters, size,
                                     kernel_initializer=initializer,
                                     strides=2,
                                     padding="same",
                                     use_bias=False))

    # batchnorm layer
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    # activation layer
    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, appy_dropout=False):
    result = tf.keras.Sequential()

    initializer = tf.random_normal_initializer(0, 0.02)

    # convolutional layer
    result.add(tf.keras.layers.Conv2DTranspose(filters, size,
                               kernel_initializer=initializer,
                               strides=2,
                               padding="same",
                               use_bias=False))

    # batchnorm layer
    result.add(tf.keras.layers.BatchNormalization())

    if appy_dropout:
        # dropout layer
        result.add(tf.keras.layers.Dropout(0.5))

    # activation layer
    result.add(tf.keras.layers.ReLU())

    return result

def Generator():
    # TODO ckeck dimensions
    inputs = tf.keras.layers.Input(shape=[None,None,3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4),                       # (bs, 64, 64, 128)
        downsample(256, 4),                       # (bs, 32, 32, 256)
        downsample(512, 4),                       # (bs, 16, 16, 512)
        downsample(512, 4),                       # (bs, 8, 8, 512)
        downsample(512, 4),                       # (bs, 4, 4, 512)
        downsample(512, 4),                       # (bs, 2, 2, 512)
        downsample(512, 4),                       # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, appy_dropout=True),      # (bs, 2, 2, 1024)
        upsample(512, 4, appy_dropout=True),      # (bs, 4, 4, 1024)
        upsample(512, 4, appy_dropout=True),      # (bs, 8, 8, 1024)
        upsample(512, 4),                         # (bs, 16, 16, 1024)
        upsample(256, 4),                         # (bs, 32, 32, 512)
        upsample(128, 4),                         # (bs, 64, 64, 256)
        upsample(64, 4),                          # (bs, 128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0, 0.02)

    last = tf.keras.layers.Conv2DTranspose(filters=3,
                           kernel_size=4,
                           strides=2,
                           padding="same",
                           kernel_initializer=initializer,
                           activation="tanh")
    # connections
    x = inputs
    s = []

    for down in down_stack:
        x = down(x)
        s.append(x)

    s = reversed(s[:-1])

    for up, sk in zip(up_stack, s):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, sk])

    last = last(x)

    return tf.keras.Model(inputs=inputs, outputs=last)



# Define el mismo generador, discriminador y optimizadores
generator = Generator()  # Define aquí tu modelo de generador
# discriminator = define_discriminator()  # Define aquí tu modelo de discriminador

# generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)

# Ruta de los checkpoints
CKPATH = "./ck"

# checkpoints = tf.io.gfile.glob(os.path.join(CKPATH, "ckpt-*"))
# checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
# last_two_checkpoints = checkpoints[-2:] if len(checkpoints) >= 2 else checkpoints
last_two_checkpoints = ['./ck/ckpt-36', './ck/ckpt-40']
checkpoint = tf.train.Checkpoint(generator=generator)

# Restaurar el último checkpoint
# latest_checkpoint = tf.train.latest_checkpoint(CKPATH)
# if latest_checkpoint:
#     print(f"Restaurando desde {latest_checkpoint}")
#     checkpoint.restore(latest_checkpoint).expect_partial()
# else:
#     print("No se encontró ningún checkpoint. Entrenamiento desde cero.")

def load_img():
    inimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file('./test_img/t.jpg'))[...,:3], tf.float32)
    inimg = tf.image.resize(inimg, [256, 256])
    inimg = (inimg / 127.5) - 1

    return tf.expand_dims(inimg, axis=0)

img = load_img()

for ckp in last_two_checkpoints:
    checkpoint.restore(ckp).expect_partial()
    out = generator(img, training=True)
    tf.keras.preprocessing.image.save_img('./test_img/out'+ ckp.split('-')[-1] + '.jpg', out[0, ...])
