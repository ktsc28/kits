import numpy as np
from skimage.transform import resize
import nibabel as nib
import pandas as pd
import cv2

def resize_image(img, target):
    return resize(img, target, order=3, mode='reflect', cval=0, clip='True', preserve_range=True, anti_aliasing=False).astype('float')


