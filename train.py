from model import load_model
from load_image import load_data
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import cv2

if __name__ == "__main__":
    data_loader = load_data()
    model = load_model((256, 256, 256, 1))
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])
    model.fit_generator(data_loader, steps_per_epoch=10)
