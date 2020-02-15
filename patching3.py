import numpy as np

class Patch:
    """Patch of image, including extra metadata
    """
    def __init__(self,xwindow,ywindow,zwindow,shift,start_coords,end_coords,patch):
        self.xwindow = xwindow
        self.ywindow = ywindow
        self.zwindow = zwindow
        self.shift = shift
        self.start_coords = start_coords
        self.end_coords = end_coords
        self.patch = patch
        self.mask = None

def patch_image3(image, mask, xwindow, ywindow, zwindow, shift, is_mask=False):
    """Assuming given a single image of an np array, image will be broken into
    the approriate number of patches based on the xlen, ylen, and the shift 
    (number of values to slide a window). Should a window exceed the dimensions 
    of the array, either x or y dims, the window will shift back until it is 
    withing the bounds of the image.
    ~~ Inputs
    :: array np.ndarray :: array which will be patched
    :: xlen int :: width of the desired patches
    :: ylen int :: height of the desired patches
    :: zlen int :: 3rd dim of the desired patches
    :: shift int :: value to slide a window over
    ~~ Outputs
    :: patches list Patch :: list of Patch objects
    """
    if not isinstance(image, np.ndarray) and len(image.shape) != 3:
        err =  "Please input an np array of dimmensions legnth 3!"
        return err
    patches = []
    dim1, dim2, dim3 = image.shape
    for zindex in range(0,dim1, shift):
        zstart = zindex
        if zwindow > dim1:
            zstart = 0
            missing_z = zwindow - dim1
            image = np.append(image, np.min(image) * np.ones((missing_z, dim2, dim3)), axis = 0)
            if is_mask == True:
                mask = np.append(image, np.zeros((missing_z, dim2, dim3)), axis = 0)
            zend = zwindow
        elif zindex + zwindow <= dim1:
            zend = zindex + zwindow
        else:
            zstart = dim1 - zwindow
            zend = dim1
        for yindex in range(0, dim2, shift):
            ystart = yindex
            if yindex + ywindow <= dim2:
                yend = yindex + ywindow
            else:
                ystart = dim2 - ywindow
                yend = dim2
            for xindex in range(0, dim3, shift):
                xstart = xindex
                if xindex + xwindow <= dim3:
                    xend = xindex + xwindow
                else:
                    xstart = dim3 - xwindow
                    xend = dim3
                pstart = [zstart,ystart,xstart]
                pend = [zend,yend,xend]
#                print("DEBUG\n",pstart,"\n",pend)
                patch = Patch(xwindow,ywindow,zwindow,shift,pstart,pend,image[pstart[0]:pend[0],pstart[1]:pend[1],pstart[2]:pend[2]])
                if is_mask == True:
                    patch.mask = mask[pstart[0]:pend[0],pstart[1]:pend[1],pstart[2]:pend[2]]
                if len(patches) > 1:
                    if np.array_equal(patch.patch, patches[-1].patch):
                #        print("DEBUG #2: Skipping because same")
                        continue
                patches.append(patch)
    return patches

if __name__ == "__main__":
    img = np.load("/home/ctadmin/data_drive/kits19/data/training/5e_i.npy")
    print(img.shape)
    patches = patch_image3(img.squeeze(), 128, 128,128, 64)
    print(len(patches))
