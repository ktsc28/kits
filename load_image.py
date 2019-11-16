import numpy as np
import nibabel as nib
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
import cv2
from preprocessing import resize_image

class DataGen(keras.utils.Sequence):
    def __init__(self, path, batch_size=1):
        self.df = pd.DataFrame(columns=['case_id', 'image', 'mask'])
        self.batch_size = batch_size
        file_list = [] 
        for root, dirs, files in os.walk(path):
            for folder in dirs:
                image_path = path + '/' + folder + "/imaging.nii.gz"
                mask_path = path + '/' + folder + "/segmentation.nii.gz"
                case_id = folder[-3:]
                self.df = self.df.append({'case_id':int(case_id), 'image':image_path, 'mask':mask_path}, ignore_index=True)
        self.df = self.df.sort_values(by=['case_id'])
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(self.df)


    def __getitem__(self, case_num, size=(128, 128, 128)):
        img = self.df.loc[self.df['case_id'] == case_num]['image']
        print("THIS IS WHAT WE ON")
        print(img)
        img = nib.load(img.values[0])
        affine = img.affine
        img = img.get_fdata()
        img = resize_image(img, size)
        img = np.expand_dims(img, axis=3)
        img = np.expand_dims(img, axis=0)
        #numpy2nifti(img, affine, 'img.nii.gz')
        mask = self.df.loc[self.df['case_id'] == case_num]['mask']
        mask = nib.load(mask.values[0])
        mask = mask.get_data()
        mask = resize_image(mask,size)
        #numpy2nifti(mask, affine, 'mask.nii.gz')
        mask = mask / 2
        mask = np.expand_dims(mask, axis=3)
        mask = np.expand_dims(mask, axis=0)
        return img, mask

    def on_epoch_end(self):
        pass

    def __len__(self):
       return 200 

def load_data():
    train_gen = DataGen("/home/kits/kits19/data/training")
    return train_gen

def numpy2nifti(np_img, affine, name):
    ni_img = nib.Nifti1Image(np_img, affine)
    nib.save(ni_img, name)

if __name__ == "__main__":
    data = DataGen("/home/kits/kits19/data/training")
    for i in range(1):
        img1, mask = data.__getitem__(i)

    #print(img.get_data_dtype() == np.dtype(np.int8))
    #print("Done")
