import numpy as np
import nibabel as nib
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from preprocessing import process, resize_image


class DataGen(tf.keras.utils.Sequence):
    def __init__(self, path, batch_size=1, is_validation=False):
        self.df = pd.DataFrame(columns=['case_id', 'image', 'mask'])
        self.batch_size = batch_size
        self.is_validation = is_validation
        file_list = [] 
        self.path = path
        for root, dirs, files in os.walk(path):
            for folder in dirs:
                image_path = path + '/' + folder + "/imaging.nii.gz"
                mask_path = path + '/' + folder + "/segmentation.nii.gz"
                case_id = folder[-3:]
                self.df = self.df.append({'case_id':int(case_id), 'image':image_path, 'mask':mask_path}, ignore_index=True)
        self.df = self.df.sort_values(by=['case_id'])
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            #print(self.df)
    
    def __getitem__(self, case_num, size=(128, 128, 128)):
        if self.is_validation == True:
            case_num += 200
        try:
            img = np.load(self.path + '/' + "{}_i.npy".format(case_num))
            mask = np.load(self.path + '/' +"{}_m.npy".format(case_num))
            return img, mask
        
        except (IOError, FileNotFoundError):
            img = self.df.loc[self.df['case_id'] == case_num]['image']
            #print("THIS IS WHAT WE ON")
            img = nib.load(img.values[0])
            affine = img.affine
            img = img.get_fdata()
            img = process(img, size)
            img = np.expand_dims(img, axis=3)
            img = np.expand_dims(img, axis=0)
            mask = self.df.loc[self.df['case_id'] == case_num]['mask']
            mask = nib.load(mask.values[0])
            mask = mask.get_data()
            #mask = resize_image(mask, size, is_mask=True)
            mask_off = mask > 1.5
            mask[mask_off] = 1  # This will turn all tumor pixels into kidney pixels
            mask = np.expand_dims(mask, axis=3)
            mask = np.expand_dims(mask, axis=0)

            np.save(self.path + '/' + "{}_i".format(case_num), img)
            np.save(self.path + '/' + "{}_m".format(case_num), mask)
            return img, mask

    def on_epoch_end(self):
        pass

    def __len__(self):
       return self.df.shape[0]

def load_data(path, is_validation=False):
    train_gen = DataGen(path, is_validation=is_validation)
    return train_gen

def numpy2nifti(np_img, affine, name):
    ni_img = nib.Nifti1Image(np_img, affine)
    nib.save(ni_img, name)

if __name__ == "__main__":
    data = DataGen("/home/ctadmin/data_drive/kits19/data/training")
    validation = DataGen("/home/ctadmin/data_drive/kits19/data/validation", is_validation=True)
    for i in range(0, 10):
        print(i)
        img1, mask = validation.__getitem__(i)    
        print("Image_size = {}    Mask_size = {}".format(img1.shape, mask.shape))

