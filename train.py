from unet import unet
from vnet import vnet
from load_image import load_data
from numba import cuda
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


if __name__ == "__main__":
    cuda.select_device(0)
    cuda.close()

    epochs = 100
    data_loader = load_data("/home/kits/kits19/data/training")
    validation_loader = load_data("/home/kits/kits19/data/validation", is_validation=True)
    
    callbacks = list()
    callbacks.append(ModelCheckpoint("weights/VNetW.h5", monitor='val_loss', save_weights_only=True, save_best_only=True))
    callbacks.append(ReduceLROnPlateau(factor=0.5, patience=2, verbose=1))
    callbacks.append(EarlyStopping(patience=2))
    
    model = vnet()
    model.fit_generator(data_loader, validation_data=validation_loader, steps_per_epoch=200, validation_steps=10, epochs=epochs, callbacks=callbacks)
    model.save_weights("weights/VNetW.h5")
