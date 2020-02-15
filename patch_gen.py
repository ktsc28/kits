import numpy as np
import os
import pandas as pd
from patching3 import patch_image3

def extract_random_patch(img, mask, size):
    z_dims = img.shape[0]
    x_dims = img.shape[1]
    y_dims = img.shape[2]
    x_limit = x_dims - size[2] - 1
    y_limit = y_dims - size[1] - 1
    if z_dims < size[0]:
        missing_slices = size[0] - img.shape[0]
        img = np.append(img, np.min(img) * np.ones((missing_slices, 512, 512)), axis=0)
        mask = np.append(mask, np.zeros((missing_slices, 512, 512), dtype=np.int8), axis=0)
        z_patch_size = size[0] 
        z_index = 0
    else:
        z_limit = z_dims - size[0]
        z_patch_size = size[0]
        z_index = np.random.randint(low=0, high=z_limit)
    x_index = np.random.randint(low=0, high=x_limit)
    y_index = np.random.randint(low=0, high=y_limit)
    patch = img[z_index:z_index+z_patch_size, x_index:x_index+size[1], y_index:y_index+size[2]]
    patch_mask = mask[z_index:z_index+z_patch_size, x_index:x_index+size[1], y_index:y_index+size[2]]
    return patch, patch_mask

def locate_kidneys(mask):
    result = np.asarray(np.where(mask > 0.1)).T

    index = np.argmax(result[:, 0])
    max_row = result[index, 0]

    index = np.argmin(result[:, 0])
    min_row = result[index, 0]

    index = np.argmax(result[:, 1])
    max_col = result[index, 1]

    index = np.argmin(result[: ,1])
    min_col = result[index, 1]

    index = np.argmax(result[:, 2])
    max_slice = result[index, 2]

    index = np.argmin(result[: ,2])
    min_slice = result[index, 2]


    return [min_row, max_row, min_col, max_col, min_slice, max_slice]

def extract_kidney_patch(img, mask, size, locations):
    x_index = np.random.randint(low=locations['start_x'], high=locations['end_x'] - size[2] / 2)
    y_index = np.random.randint(low=locations['start_y'], high=img.shape[1] - size[1])
    if img.shape[0] - locations['end_z'] + 1 > size[0]:
        z_index = np.random.randint(low=locations['start_z'], high=locations['end_z']) 
        z_end = z_index + size[0]
    elif locations['start_z'] + size[0] < img.shape[0]:
        z_index = np.random.randint(low=locations['start_z'], high=img.shape[0] - size[0])
        z_end = z_index + size[0]
    else:
        z_index = 0
        if img.shape[0] < size[0]:
            missing_slices = size[0] - img.shape[0]
            img = np.append(img, np.min(img) * np.ones((missing_slices, 512, 512)), axis=0)
            mask = np.append(mask, np.zeros((missing_slices, 512, 512), dtype=np.int8), axis=0)
    patch = img[z_index:z_index + size[0], y_index:y_index + size[1], x_index:x_index+size[2]]
    patch_mask = mask[z_index:z_index + size[0], y_index:y_index + size[1], x_index:x_index+size[2]]
    return patch, patch_mask

def generate_patches(path, is_validation=False):
    if is_validation == False:
        kidney_locations = pd.read_csv("kidney_location_data.csv")
        num_images = 200
        start = 0
        end = 200
    else:
        kidney_locations = pd.read_csv("kidney_location_data_validation.csv")
        num_images = 10
        start = 200
        end = 210
    for i in range(start, end):
        img = np.load(path + "/" + "{}_i.npy".format(i))
        mask = np.load(path + "/" + "{}_m.npy".format(i))
        img = np.squeeze(img)
        mask = np.squeeze(mask)
        for j in range(4):
            if j == 0:
                patch, patch_mask = extract_random_patch(img, mask, (128, 128, 128))
            else:
                patch, patch_mask = extract_kidney_patch(img, mask, (128, 128, 128), kidney_locations.loc[i-start,:])
            patch = np.expand_dims(patch, axis=3)
            patch = np.expand_dims(patch, axis=0)
            patch_mask = np.expand_dims(patch_mask, axis=3)
            patch_mask = np.expand_dims(patch_mask, axis=0)
            if is_validation == False:
                np.save("/home/kits/kits19/data/training_patches/{}_i_patch_{}".format(i, j), patch)
                np.save("/home/kits/kits19/data/training_patches/{}_m_patch_{}".format(i, j), patch_mask)
            else:
                np.save("/home/kits/kits19/data/validation_patches/{}_i_patch_{}".format(i, j), patch)
                np.save("/home/kits/kits19/data/validation_patches/{}_m_patch_{}".format(i, j), patch_mask)

def whole_image_patches(path, is_validation=False):
    if is_validation == True:
        start = 200
        end = 210
    else:
        start = 150
        end = 200   
    for i in range(start, end):
        img = np.load(path + "/" + "{}_i.npy".format(i)).squeeze()
        mask = np.load(path + "/" + "{}_m.npy".format(i)).squeeze()
        patch_list = patch_image3(img, mask, 128, 128, 128, 64, is_mask=True)
        j = 0
        print("Image: {}, shape: {}, num_patches: {}".format(i, img.shape, len(patch_list)))
        for patch in patch_list:
            if patch.patch.shape != (128, 128, 128) or patch.mask.shape != (128, 128, 128):
                print(patch.patch.shape)
                print("FUCKING ERROR")
            patch.patch = np.expand_dims(patch.patch, axis=3)
            patch.patch = np.expand_dims(patch.patch, axis=0)
            patch.mask = np.expand_dims(patch.mask, axis=3)
            patch.mask = np.expand_dims(patch.mask, axis=0)
            if is_validation == False:
                np.save("/home/ctadmin/data_drive/kits19/data/training_patches4/{}_i_patch_{}".format(i, j), patch.patch)
                np.save("/home/ctadmin/data_drive/kits19/data/training_patches4/{}_m_patch_{}".format(i, j), patch.mask)
            else:
                np.save("/home/ctadmin/data_drive/kits19/data/validation_patches/{}_i_patch_{}".format(i, j), patch.patch)
                np.save("/home/ctadmin/data_drive/kits19/data/validation_patches/{}_m_patch_{}".format(i, j), patch.mask)
            j += 1


if __name__ == "__main__":
    locations_data = pd.DataFrame(columns=["case_id", "start_z", "end_z", "start_y", "end_y", "start_x", "end_x"])
   # data_generator = load_data("/home/kits/kits19/data/training", is_validation=False)
    #generate_patches("/home/ctadmin/data_drive/kits19/data/training/", is_validation=False)
    whole_image_patches("/home/ctadmin/data_drive/kits19/data/validation/", is_validation=True)
    #for i in range(10):
    #    img, mask = data_generator.__getitem__(i)
    #    coords = locate_kidneys(mask.squeeze())
    #    print("ID: {}, Coords: {}".format(i, coords))
    #    locations_data = locations_data.append({"case_id":i, "start_z":coords[0], "end_z":coords[1], \
    #        "start_y":coords[2], "end_y":coords[3], "start_x":coords[4], "end_x":coords[5]}, ignore_index=True)
    #locations_data.to_csv("kidney_location_data_validation.csv") 

