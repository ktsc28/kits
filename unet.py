import tensorflow as tf 
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam


def unet(input_shape=(128, 128, 128, 1), optimizer=Adam(lr=0.0005),
         loss='binary_crossentropy', metrics=["accuracy"], batch_size=1):
    input_layer = Input(shape=input_shape, batch_size=batch_size)
    conv_1 = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(input_layer)
    conv_2 = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(conv_1)
    pool_1 = MaxPool3D((2, 2, 2), strides=None, data_format='channels_last')(conv_2)

    conv_3 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(pool_1)
    conv_4 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(conv_3)
    pool_2 = MaxPool3D((2, 2, 2), strides=(2,2,2), data_format='channels_last')(conv_4)
    
    conv_5 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(pool_2)
    conv_6 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(conv_5)
    pool_3 = MaxPool3D((2, 2, 2), strides=(2,2,2), data_format='channels_last')(conv_6)

    # Bottleneck
    conv_7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(pool_3)
    conv_8 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(conv_7)

    up_1 = UpSampling3D((2, 2, 2), data_format='channels_last')(conv_8)
    concat = Concatenate()([up_1, conv_6])
    conv_9 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(concat)
    conv_10 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(conv_9)

    up_2 = UpSampling3D((2, 2, 2), data_format='channels_last')(conv_10)
    concat = Concatenate()([up_2, conv_4])
    conv_11 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(concat)
    conv_12 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(conv_11)

    up_3 = UpSampling3D((2, 2, 2), data_format='channels_last')(conv_12)
    concat = Concatenate()([up_3, conv_2])
    conv_13 = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(concat)
    conv_14 = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(conv_13)

    output_layer = Conv3D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid', data_format='channels_last')(conv_14)

    model = Model(input_layer, output_layer)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


if __name__ == "__main__":
    model = unet()
    model.summary()
