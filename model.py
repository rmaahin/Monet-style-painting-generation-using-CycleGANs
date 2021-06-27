import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

orig_img_size = (256, 256)
input_img_size = (256, 256, 3)
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

class ReflectionPadding2D(layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)
    
    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")

def residual_block(
    x,
    activation,
    kernel_initializer = kernel_init,
    kernel_size = (3, 3),
    strides = (1, 1),
    padding = "valid",
    gamma_initializer = gamma_init,
    use_bias = False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides = strides,
        kernel_initializer = kernel_initializer,
        padding = padding,
        use_bias = use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer = gamma_initializer)(x)
    x = activation(x)
    
    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides = strides,
        kernel_initializer = kernel_initializer,
        padding = padding,
        use_bias = use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer = gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x

def downsample(
    x, 
    filters,
    activation,
    kernel_initializer = kernel_init,
    kernel_size = (3, 3),
    strides = (2, 2),
    padding = "same",
    gamma_initializer = gamma_init,
    use_bias = False
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides = strides,
        kernel_initializer = kernel_initializer,
        padding = padding,
        use_bias = use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer = gamma_initializer)(x)
    if activation:
        x = activation(x)
        
    return x

def upsample(
    x,
    filters,
    activation,
    kernel_initializer = kernel_init,
    kernel_size = (3, 3),
    strides = (2, 2),
    padding = "same",
    gamma_initializer = gamma_init,
    use_bias = False
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides = strides,
        padding = padding,
        kernel_initializer = kernel_initializer,
        use_bias = use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer = gamma_initializer)(x)
    if activation:
        x = activation(x)
        
    return x

def get_resnet_generator(
    filters = 64,
    num_downsampling_blocks = 2,
    num_residual_blocks = 9,
    num_upsampling_blocks = 2,
    gamma_initializer = gamma_init,
    name = None,
):
    img_input = layers.Input(shape = input_img_size, name = name+"_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation('relu')(x)
    
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation('relu'))
        
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation('relu'))
        
    for _ in range(num_upsampling_blocks):
        filters //= 2
        x = upsample(x, filters=filters, activation=layers.Activation('relu'))
        
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation('tanh')(x)
    
    model = keras.models.Model(img_input, x, name=name)
    return model

def get_discriminator(
    filters=64, kernel_initializer=kernel_init, num_downsampling=3, name=None
):
    img_input = layers.Input(shape=input_img_size, name=name+"_img_input")
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides = (2, 2),
        padding = "same",
        kernel_initializer = kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)
    
    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )
        
    x = layers.Conv2D(
        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    )(x)
    
    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model