import numpy as np
import nibabel as nib
import pandas as pd
import os
#import tensorflow as tf
from tensorflow import keras
from preprocessing import process, resize_image, random_flips, random_image_processing
from random import random
from patch_gen import generate_patches


class DataGen(keras.utils.Sequence):
    def __init__(self, paths, batch_size=1, is_validation=False):
        self.df = pd.DataFrame(columns=['case_id', 'image', 'mask'])
        self.batch_size = batch_size
        self.is_validation = is_validation
        file_list = [] 
        self.paths = paths
        for path in paths:
            for root, dirs, files in os.walk(path):
                #for folder in dirs:
                #    image_path = path + '/' + folder + "/imaging.nii.gz"
                #    mask_path = path + '/' + folder + "/segmentation.nii.gz"
                #    case_id = folder[-3:]
                #    self.df = self.df.append({'case_id':int(case_id), 'image':image_path, 'mask':mask_path}, ignore_index=True)
                for file in files:
                    ext = os.path.splitext(file)[-1].lower()
                    if ext == '.npy':
                        s = file.split('_')
                        if s[1] == 'i':
                            image_path = path + '/' + file
                            s[1] = 'm'
                            mask_path = path + '/' + "_".join(s)
                            self.df = self.df.append({'case_id':int(s[0]), 'image':image_path, 'mask':mask_path}, ignore_index=True)
        print(len(self.df))
        self.df = self.df.sort_values(by=['case_id'])
        self.iter = 1

        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            #print(self.df)
    
    def __getitem__(self, case_num, size=(128, 128, 128)):
        img = self.df.iloc[[case_num]]['image']
        img = np.load(img.values[0])
        #print("THIS IS WHAT WE ON")
        #img = process(img, size)
        #img = resize_image(img, size)
        #img = np.squeeze(np.array(img))
        #img = np.transpose(img)
        #img = np.flip(img, axis=1)

        mask = self.df.iloc[[case_num]]['mask']
        mask = np.load(mask.values[0])
        #mask = resize_image(mask, size, is_mask=True)
        #numpy2nifti(mask, affine, 'mask.nii.gz')
        #mask = mask / 2
        #mask_off = mask > 1.5
        #mask[mask_off] = 1
        #mask = np.expand_dims(mask, axis=3)
        #mask = np.expand_dims(mask, axis=0)
        #p = random()
        #img = random_flips(img, p)
        #mask = random_flips(mask, p)
        return img, mask

    def on_epoch_end(self):
        generate_patches("/home/ctadmin/data_drive/kits19/data/training/", is_validation=False)
        generate_patches("/home/ctadmin/data_drive/kits19/data/validation/", is_validation=True)

    def __len__(self):
        if self.is_validation == False:
            self.iter += 1
            if self.iter % 2 == 0:
                generate_patches("/home/ctadmin/data_drive/kits19/data/training/", is_validation=False)
                generate_patches("/home/ctadmin/data_drive/kits19/data/validation/", is_validation=True)
        return self.df.shape[0]

def load_data(path, is_validation=False):
    train_gen = DataGen(path, is_validation=is_validation)
    return train_gen

def numpy2nifti(np_img, affine, name):
    ni_img = nib.Nifti1Image(np_img, affine)
    nib.save(ni_img, name)

if __name__ == "__main__":
    data = DataGen(["/home/ctadmin/data_drive/kits19/data/training_patches", "/home/ctadmin/data_drive/kits19/data/training_patches2"])
    #validation = DataGen("/home/kits/kits19/data/validation_patches", is_validation=True)
    for i in range(1, 2):
        #img, mask = validation.__getitem__(i)    
        img, mask = data.__getitem__(i)
        #numpy2nifti(img.squeeze(), np.eye(4), "test")
        #numpy2nifti(mask.squeeze(), np.eye(4), "test_mask")


