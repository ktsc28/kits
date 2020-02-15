from unet import unet
from vnet import vnet
from load_image import load_data
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
import tensorflow as tf
import pickle


def save_history(history):
    with open('history.db', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


if __name__ == "__main__":
    epochs = 50
    data_loader = load_data("/home/ctadmin/data_drive/kits19/data/training_patches")
    validation_loader = load_data("/home/kits/kits19/data/validation_patches", is_validation=True)
    
    callbacks = list()
    callbacks.append(ModelCheckpoint("best_VNetW.h5", monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=1))
    # callbacks.append(ReduceLROnPlateau(factor=0.5, patience=2, verbose=1))
    callbacks.append(EarlyStopping(monitor='val_loss', patience=10))
    callbacks.append(CSVLogger("training.log", append=True))
    
    model = unet()
    history = model.fit_generator(data_loader, validation_data=validation_loader, epochs=epochs, callbacks=callbacks)
    model.save_weights("weights/VNetW.h5")
    save_history(history)
