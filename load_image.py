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
                self.df = self.df.append({'case_id':int(case_id), 'image':image_path, 'mask':mask_path}, ignore_index=True)
        self.df = self.df.sort_values(by=['case_id'])
        print(self.df)



    def __getitem__(self, case_num):
        img = self.df.loc[self.df['case_id'] == case_num]['image']
        img = nib.load(img.values[0])
        img = img.get_fdata()
        print(type(img))
        img = np.append(img, np.zeros((512, 512)), 1)
        mask = self.df.loc[self.df['case_id'] == case_num]['mask']
        mask = nib.load(mask.values[0])
        mask = mask.get_data()
        mask = mask / 2
        return img, mask

    def on_epoch_end(self):
        pass

    def __len__(self):
       return 300 

def load_data():
    train_gen = DataGen("/home/kits/kits19/data")
    return train_gen


if __name__ == "__main__":
    data = DataGen("/home/kits/kits19/data")
    img1, mask = data.__getitem__(1)

    print(img1.shape)
    #print(img.get_data_dtype() == np.dtype(np.int8))
    #print("Done")
