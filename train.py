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
    data_loader = load_data("/srv/kits19/data/training")
    validation_loader = load_data("/srv/kits19/data/validation", is_validation=True)
    
    callbacks = list()
    callbacks.append(ModelCheckpoint("best_VNetW.h5", monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=1))
    # callbacks.append(ReduceLROnPlateau(factor=0.5, patience=2, verbose=1))
    callbacks.append(EarlyStopping(monitor='val_loss', patience=10))
    callbacks.append(CSVLogger("training.log", append=True))
    
    model = vnet()
    history = model.fit_generator(data_loader, validation_data=validation_loader, steps_per_epoch=200, validation_steps=10, epochs=epochs, callbacks=callbacks)
    model.save_weights("weights/VNetW.h5")
    save_history(history)
