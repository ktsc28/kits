import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def downward_layer(input_layer, n_convolutions, n_output_channels):
    inl = input_layer
    for _ in range(n_convolutions):
        inl = PReLU()(
            Conv3D(filters=(n_output_channels // 2), kernel_size=5,
                   padding='same', kernel_initializer='he_normal')(inl)
        )
    add_l = add([inl, input_layer])
    downsample = Conv3D(filters=n_output_channels, kernel_size=2, strides=2,
                        padding='same', kernel_initializer='he_normal')(add_l)
    downsample = PReLU()(downsample)
    return downsample, add_l


def upward_layer(input0, input1, n_convolutions, n_output_channels):
    merged = concatenate([input0, input1], axis=4)
    inl = merged
    for _ in range(n_convolutions):
        inl = PReLU()(
            Conv3D((n_output_channels * 4), kernel_size=5,
                   padding='same', kernel_initializer='he_normal')(inl)
        )
    add_l = add([inl, merged])
    upsample = Conv3DTranspose(filters=n_output_channels, kernel_size=2, padding='same', strides=2)(add_l)
    return PReLU()(upsample)


def vnet(input_size=(128, 128, 128, 1), optimizer=Adam(lr=1e-5),
         loss='binary_crossentropy', metrics=['accuracy']):
    
    # Layer 1
    inputs = Input(input_size)
    conv_1 = Conv3D(16, kernel_size=5, strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    conv_1 = PReLU()(conv_1)
    repeat_1 = concatenate(16 * [inputs], axis=-1)
    add_1 = add([conv_1, repeat_1])
    down_1 = Conv3D(32, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(add_1)
    down_1 = PReLU()(down_1)

    # Layer 2,3,4
    down_2, add_2 = downward_layer(down_1, 2, 64)
    down3, add_3 = downward_layer(down_2, 3, 128)
    down_4, add_4 = downward_layer(down3, 3, 256)

    # Layer 5
    conv_5_1 = Conv3D(256, kernel_size=5, strides=1, padding='same', kernel_initializer='he_normal')(down_4)
    conv_5_1 = PReLU()(conv_5_1)
    conv_5_2 = Conv3D(256, kernel_size=5, strides=1, padding='same', kernel_initializer='he_normal')(conv_5_1)
    conv_5_2 = PReLU()(conv_5_2)
    conv_5_3 = Conv3D(256, kernel_size=5, strides=1, padding='same', kernel_initializer='he_normal')(conv_5_2)
    conv_5_3 = PReLU()(conv_5_3)
    add_5 = add([conv_5_3, down_4])
    upsample_5 = Conv3DTranspose(128, kernel_size=2, strides=2, padding='same')(add_5)
    upsample_5 = PReLU()(upsample_5)

    # Layer 6,7,8
    upsample_6 = upward_layer(upsample_5, add_4, 3, 64)
    upsample_7 = upward_layer(upsample_6, add_3, 3, 32)
    upsample_8 = upward_layer(upsample_7, add_2, 2, 16)

    # Layer 9
    merged_9 = concatenate([upsample_8, add_1], axis=4)
    conv_9_1 = Conv3D(32, kernel_size=5, strides=1, padding='same', kernel_initializer='he_normal')(merged_9)
    conv_9_1 = PReLU()(conv_9_1)
    add_9 = add([conv_9_1, merged_9])
    conv_9_2 = Conv3D(1, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(add_9)
    conv_9_2 = PReLU()(conv_9_2)

    # Output layer
    sigmoid = Conv3D(1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                     activation='sigmoid')(conv_9_2)
    #softmax = Softmax()(conv_9_2)

    model = Model(inputs=inputs, outputs=sigmoid)
    model.compile(optimizer, loss, metrics)

    return model


if __name__ == "__main__":
    model = vnet((128,128,64,1))
    model.summary(line_length=113)
