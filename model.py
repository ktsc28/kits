import numpy as np
import nibabel as nib
import tensorflow as tf 

def load_model(input_shape):
    input_layer = tf.keras.layers.Input(shape=input_shape, batch_size=1)
    conv_1 = tf.keras.layers.Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(input_layer)
    conv_2 = tf.keras.layers.Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(conv_1)
    pool_1 = tf.keras.layers.MaxPool3D((2, 2, 2), strides=None, data_format='channels_last')(conv_2)

    conv_3 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(pool_1)
    conv_4 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(conv_3)
    pool_2 = tf.keras.layers.MaxPool3D((2, 2, 2), strides=(2,2,2), data_format='channels_last')(conv_4)
    
    conv_5 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(pool_2)
    conv_6 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(conv_5)
    pool_3 = tf.keras.layers.MaxPool3D((2, 2, 2), strides=(2,2,2), data_format='channels_last')(conv_6)

    # Bottleneck
    conv_7 = tf.keras.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(pool_3)
    conv_8 = tf.keras.layers.Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(conv_7)

    up_1 = tf.keras.layers.UpSampling3D((2, 2, 2), data_format='channels_last')(conv_8)
    concat = tf.keras.layers.Concatenate()([up_1, conv_6])
    conv_9 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(concat)
    conv_10 = tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(conv_9)

    up_2 = tf.keras.layers.UpSampling3D((2, 2, 2), data_format='channels_last')(conv_10)
    concat = tf.keras.layers.Concatenate()([up_2, conv_4])
    conv_11 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(concat)
    conv_12 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(conv_11)

    up_3 = tf.keras.layers.UpSampling3D((2, 2, 2), data_format='channels_last')(conv_12)
    concat = tf.keras.layers.Concatenate()([up_3, conv_2])
    conv_13 = tf.keras.layers.Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(concat)
    conv_14 = tf.keras.layers.Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', data_format='channels_last')(conv_13)

    output_layer = tf.keras.layers.Conv3D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid', data_format='channels_last')(conv_14)

    return tf.keras.models.Model(input_layer, output_layer)


if __name__ == "__main__":
    #img = nib.load('kits19\data\case_00000\imaging.nii.gz')
    #shape = img.dataobj.shape
    model = load_model((128, 128, 128, 1))
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
