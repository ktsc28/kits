import numpy as np
import nibabel as nib
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras

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
                self.df = self.df.append({'case_id':case_id, 'image':image_path, 'mask':mask_path}, ignore_index=True)

    def __getitem__(self, case_num):
        img = nib.load(self.df['image'][case_num])
        img = img.get_fdata()
        mask = nib.load(self.df['mask'][case_num])
        mask = mask.get_fdata()
        mask = mask / 2
        return img, mask

    def on_epoch_end(self):
        pass

    def __len__(self):
       return 300 

def load_data():
    train_gen = DataGen("./kits19/data")
    return train_gen


if __name__ == "__main__":
    data = DataGen("./kits19/data")
    img = data.__getitem__(0)
    #img = nib.load('kits19\data\case_00000\imaging.nii.gz')
    #img1 = img.get_fdata()
    #print(img.get_data_dtype() == np.dtype(np.int8))
    #print("Done")
    print('fd')