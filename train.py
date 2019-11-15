from model import load_model
from load_image import load_data
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import cv2

if __name__ == "__main__":
    data_loader = load_data()
    img, mask = data_loader.__getitem__(0)
    pass
