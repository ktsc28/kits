import numpy as np
from skimage.transform import resize
import nibabel as nib
import pandas as pd
import tensorflow as tf

def resize_image(img, target, is_mask=False):
    t = img.dtype
    return resize(img, target, order=3, mode='reflect', cval=0, clip='True', preserve_range=True, anti_aliasing=True).astype(t)

def normalize(img):
    low_bound = np.percentile(img, 0.5)
    upper_bound = np.percentile(img, 99.5)
    img = np.clip(img, low_bound, upper_bound)
    img = (img - img.mean()) / img.std()
    return img

def process(img, target):
    #img = resize_image(img, target)
    img = normalize(img)
    return img

def random_flips(img, p):
    if p < 0.33:
        img = np.fliplr(img)
    elif p >= 0.33 and p < 0.66:
        img = np.flipud(img)
    return img

def random_image_processing(x):
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x
