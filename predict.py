import numpy as np
import tensorflow as tf
from load_image import load_data,numpy2nifti
from vnet import vnet
from preprocessing import resize_image


def predict():
    model = vnet()
    model.load_weights("weights/VNetW.h5")
    
    loader = load_data("/home/kits/kits19/data/validation/", is_validation=True)
    img, _ = loader.__getitem__(0)
    mask = model.predict(img)
    
    mask = np.squeeze(np.array(mask))
    img = np.squeeze(np.array(img))

    max_val =np.max(mask)
    mid = max_val * 0.40
    mask_off = mask < mid
    mask[mask_off] = 0
    mask_on =mask >= mid
    mask[mask_on] = 1

    mask = np.transpose(mask)
    mask = np.flip(mask, axis=1)
    img = np.transpose(img)
    img = np.flip(img, axis=1)
    
    
    numpy2nifti(mask, np.eye(4), 'prediction.nii.gz')
    numpy2nifti(img, np.eye(4), 'img.nii.gz')
    return


if __name__ == "__main__":
    predict()
