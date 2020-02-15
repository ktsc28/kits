import numpy as np
import tensorflow as tf
from load_image import load_data,numpy2nifti
from vnet import vnet
from unet import unet
from preprocessing import resize_image
import nibabel as nib
from patching3 import patch_image3


def predict():
    model = unet()
    model.load_weights("weights/VNetW.h5")
    
    #loader = load_data("/home/kits/kits19/data/validation_patches", is_validation=True)
    for i in range(1):
        #img, real_mask = loader.__getitem__(i)
        img = np.load("/home/ctadmin/data_drive/kits19/data/validation/200_i.npy")
        real_mask = np.load("/home/ctadmin/data_drive/kits19/data/validation/200_m.npy")
        patch_list = patch_image3(img.squeeze(), None, 128, 128, 128, 64)
        img = img.squeeze()
        full_mask = np.zeros(img.shape)
        for patch in patch_list:
            patch.patch = np.expand_dims(patch.patch, axis=3)
            patch.patch = np.expand_dims(patch.patch, axis=0)
            partial_mask = np.copy(model.predict(patch.patch)).squeeze()
            z = patch.start_coords[0]
            y = patch.start_coords[1]
            x = patch.start_coords[2]
            window_size = patch.xwindow
            z_stop = min(window_size, img.shape[0])

            max_val = np.max(partial_mask)
            mid = 0.9 * max_val
            mask_off = partial_mask < mid
            partial_mask[mask_off] = 0
            mask_on = partial_mask >= mid
            partial_mask[mask_on] = 1

            full_mask[z:z+z_stop, y:y+window_size, x:x+window_size]= (full_mask[z:z+z_stop, y:y+window_size, x:x+window_size] + np.copy(partial_mask[:z_stop,:,:])) / 2
    
        #real_mask = nib.load("/home/ctadmin/kits/kits/mask_good_kidney_scan.nii.gz")
        #real_mask = real_mask.get_fdata()
        real_mask = np.expand_dims(real_mask, axis=3)
        real_mask = np.expand_dims(real_mask, axis=0)

        #mask = model.predict(img)
        
        full_mask = np.squeeze(np.array(full_mask))
        real_mask = np.squeeze(np.array(real_mask))

        max_val = np.max(full_mask)
        mid = 0.9 * max_val
        mask_off = full_mask < mid
        full_mask[mask_off] = 0
        mask_on = full_mask >= mid
        full_mask[mask_on] = 1

        full_mask = np.transpose(full_mask)
        full_mask = np.flip(full_mask, axis=1)
        real_mask = np.transpose(real_mask)
        real_mask = np.flip(real_mask, axis=1)
        img = np.transpose(img)
        img = np.flip(img, axis=1)
        
        
        numpy2nifti(real_mask, np.eye(4), 'ground_truth{}.nii.gz'.format(i))
        numpy2nifti(full_mask, np.eye(4), 'prediction{}.nii.gz'.format(i))
        numpy2nifti(img, np.eye(4), 'img{}.nii.gz'.format(i))
    return


if __name__ == "__main__":

    predict()

