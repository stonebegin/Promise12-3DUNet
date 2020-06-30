import h5py
import numpy as np
import os
import glob
import nibabel as nib
import SimpleITK as sitk
import cv2
from skimage.exposure import equalize_adapthist

image_path = '/home/stonebegin/Workspace/datasets/promise12/Images/'
mask_path = '/home/stonebegin/Workspace/datasets/promise12/Masks/'

def img_resize(imgs, img_rows, img_cols, equalize=True):
    
    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist( img, clip_limit=0.05 )
            # img = clahe.apply(cv2.convertScaleAbs(img))

        new_imgs[mm] = cv2.resize( img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST )

    return new_imgs

def crop(imgs, equalize=True):
    d, w, h = imgs.shape()
    quarter = d // 4
    half_d = d // 2
    half_w = w // 2
    half_h = h // 2
    # new_imgs = np.zeros([16, 256, 256])
    for img in imgs:
        if equalize:
            img = equalize_adapthist(img, clip_limit=0.05)
    if d >= 16 * 2:
        return [imgs[quarter-8 : quarter+8, half_w-128 : half_w+128, half_h-128 : half_h+128], imgs[d-1-quarter-8 : d-1-quarter+8, half_w-128 : half_w+128, half_h-128 : half_h+128]]
    else:
        return imgs[half_d-8 : half_d+8, half_w-128 : half_w+128, half_h-128 : half_h+128]

def save_data(filename, image, mask):
    f = h5py.File('./train_data/' + filename + '.h5', 'w')
    f['raw'] = image
    f['label'] = mask
    f.close()

image_list = sorted(os.listdir(image_path))
mask_list = sorted(os.listdir(mask_path))
print(image_list)
print(mask_list)

Images = {}
Masks = {}
validation_percentage = 0.1
Images['validation'] = image_list[0:int(len(image_list) * validation_percentage)]
Images['train'] = image_list[int(len(image_list) * validation_percentage) : len(image_list)]

Masks['validation'] = mask_list[0:int(len(mask_list) * validation_percentage)]
Masks['train'] = mask_list[int(len(mask_list) * validation_percentage) : len(mask_list)]

print(Images, Masks)

print('stage training...')
IDs = 0
for filename in Images['train']:
    print(os.path.isfile(image_path + filename))
    itkimage = sitk.ReadImage(image_path + filename)
    img = sitk.GetArrayFromImage(itkimage)
    img = np.array(img)
    # img = np.rollaxis(img, 3, 0)
    print('image shape: ', img.shape)
    itkimage = sitk.ReadImage(mask_path + filename.split('.')[0] + '_segmentation.mhd')
    mask = sitk.GetArrayFromImage(itkimage)
    mask = np.array(mask)
    print('mask shape: ', mask.shape)

    new_image = crop(img)
    new_mask = crop(mask)
    if len(new_image) == 2:
        save_data(filename.split('.')[0] + '_' + IDs, new_image[0], new_mask[0])
        IDs += 1
        save_data(filename.split('.')[0] + '_' + IDs, new_image[1], new_image[1])
        IDs += 1
    else:
        save_data(filename.split('.')[0] + '_' + IDs, new_image[0], new_mask[0])
        IDs += 1