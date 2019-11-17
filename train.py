from model import load_model
from load_image import load_data
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import cv2

if __name__ == "__main__":
    epochs = 1
    data_loader = load_data("/home/kits/kits19/data/training")
    validation_loader = load_data("/home/kits/kits19/data/validation", is_validation=True)
    model = load_model((128, 128, 128, 1))
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])
    model.fit_generator(data_loader, validation_data=validation_loader, steps_per_epoch=200, validation_steps=10, epochs=epochs)
    model.save_weights("UNetW.h5")
