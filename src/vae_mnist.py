# -------------------------------
# Imports
# -------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import norm
from keras import backend, metrics, callbacks
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Input, Lambda, Layer, Reshape
from keras.datasets import mnist
from keras.utils import plot_model
from keras.utils import plot_model

backend.clear_session()

# -------------------------------
# Model Parameters and Dataset
# -------------------------------

img_shape = (28, 28, 1)
batch_size = 16
latent_dim = 10     # change this for different plots
epochs = 50

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.astype('float32') / 255
x_test = x_test.reshape(x_test.shape + (1,))

# -------------------------------
# Custom Layers
# -------------------------------

def sampling(args):
    z_mean, z_log_var = args
    epsilon = backend.random_normal(shape=(backend.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
    return z_mean + backend.exp(z_log_var) * epsilon

class VariationalLayer(Layer):
    def vae_loss(self, x, z_decoded):
        x = backend.flatten(x)
        z_decoded = backend.flatten(z_decoded)
        xent_loss = metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * backend.mean(
            1 + z_log_var - backend.square(z_mean) - backend.exp(z_log_var), axis=-1)
        return backend.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

# -------------------------------
# Encoder Architecture
# -------------------------------

image = Input(shape=img_shape)

x = Conv2D(32, 3, padding='same', activation='relu')(image)
x = Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)

shape_before_flattening = backend.int_shape(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)

z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)
z = Lambda(sampling)([z_mean, z_log_var])
encoder = Model(image, z_mean)

# plot encoder
plot_model(encoder,  to_file='encoder.png')


# -------------------------------
# Decoder Architecture
# -------------------------------

decoder_input = Input(backend.int_shape(z)[1:])

x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
x = Reshape(shape_before_flattening[1:])(x)
x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = Conv2D(1, 3, padding='same', activation='sigmoid')(x)

decoder = Model(decoder_input, x)
z_decoded = decoder(z)
y = VariationalLayer()([image, z_decoded])

# plot decoder
plot_model(decoder,  to_file='decoder.png')


# -------------------------------
# Model Training
# -------------------------------

vae = Model(image, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()
history = vae.fit(x=x_train, y=None, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))

# plot history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# -------------------------------
# Plot Data
# -------------------------------

# 2D Latent Space
if (latent_dim == 2):    
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.show()

# Manifold of Digits
if (latent_dim == 2):
    n = 17
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
            x_decoded = decoder.predict(z_sample, batch_size=batch_size)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()



