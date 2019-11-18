import numpy as np
from skimage.transform import resize
import nibabel as nib
import pandas as pd
import cv2

def resize_image(img, target, is_mask=False):
    t = img.dtype
    return resize(img, target, order=3, mode='reflect', cval=0, clip='True', preserve_range=True, anti_aliasing=False).astype(t)

def normalize(img):
    low_bound = np.percentile(img, 0.5)
    upper_bound = np.percentile(img, 99.5)
    img = np.clip(img, low_bound, upper_bound)
    img = (img - img.mean()) / img.std()
    return img

def process(img, target):
    img = resize_image(img, target)
    img = normalize(img)
    return img





