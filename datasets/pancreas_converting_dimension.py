import nibabel as nib
import numpy as np
from PIL import Image
from skimage import exposure, color
import matplotlib.pyplot as plt

#load the image
img = nib.load("../data/imagesTr_3d/pancreas_001.nii") #just preview the voxels are not really loaded yet
data = img.get_fdata() # loading data in memory

#get number of slices along each axis
slices_x, slices_y, slices_z = data.shape

for i in range(slices_z):
    slice = data[:, :, i] #all x and y slices for one certain z_slice
    img = Image.fromarray(np.uint8(slice))
    img.save("../data/imagesTr_2d/pancreas_001_slice{}.jpg".format(i))

#%%
#load the labels
label = nib.load("../data/labelsTr_3d/pancreas_001.nii")
data = label.get_fdata()

# displaying one slice of the image
mid_slice_z = data[:, :, 50]
plt.imshow(mid_slice_z.T, cmap='summer')
plt.show()
#numpy.savez und als numpy array speichern
#slices mit keinen labeln rausscheißen

slices_x, slices_y, slices_z = data.shape
for i in range(slices_z):
    slice = data[:, :, i] #all x and y slices for one certain z_slice
    #setting the cmap of slice to summer
    slice = exposure.rescale_intensity(slice, out_range=(0, 255)).astype(np.uint8)
    #label = Image.fromarray(slice)
    label = color.label2rgb(slice, bg_label=0, kind='avg')
    label = Image.fromarray(np.uint8(label))
    #label = color.gray2rgb(label)
    #label = Image.fromarray(label)
    #print("color space: ",label.getcolors())
    label.save("../data/labelsTr_2d/pancreas_001_slice.jpg")

