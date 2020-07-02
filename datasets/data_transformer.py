import h5py
import numpy as np
import os
import glob
import nibabel as nib
import SimpleITK as sitk
import cv2
from skimage.exposure import equalize_adapthist

def crop(imgs, equalize=True):
    imgs = np.array(imgs)
    d, w, h = imgs.shape
    quarter = d // 4
    half_d = d // 2
    half_w = w // 2
    half_h = h // 2

    for img in imgs:
        if equalize:
            img = equalize_adapthist(img, clip_limit=0.05)
    if d < 16:
        new_imgs = np.zeros([16, 256, 256])
        new_imgs[16-d:, :, :] = imgs[half_d-8 : half_d+8, half_w-128 : half_w+128, half_h-128 : half_h+128]
        return new_imgs
    if d >= 16 * 2:
        return [imgs[quarter-8 : quarter+8, half_w-128 : half_w+128, half_h-128 : half_h+128], imgs[d-1-quarter-8 : d-1-quarter+8, half_w-128 : half_w+128, half_h-128 : half_h+128]]
    else:
        return imgs[half_d-8 : half_d+8, half_w-128 : half_w+128, half_h-128 : half_h+128]

def save_data(filename, image, mask):
    print(filename, image.shape, mask.shape)
    f = h5py.File('../data/' + filename + '.h5', 'w')
    f['raw'] = image
    f['label'] = mask
    f.close()

def process_imgs(file_list):
    IDs = 0
    for path_name in file_list:
        root = path_name
        paths = sorted(os.listdir(root))
        print(paths)
    
        for idx, path in enumerate(paths):
            if path.find('mhd')>=0 and path.find('segm') < 0:
                data = sitk.ReadImage(os.path.join(root, path))
                segmentation = sitk.ReadImage(os.path.join(root, paths[idx+2]))
                print(path, paths[idx+2])

                image = sitk.GetArrayFromImage(data)
                mask = sitk.GetArrayFromImage(segmentation)


                new_image = crop(image)
                new_mask = crop(mask)
                # print(new_mask)
                if len(new_image) == 2:
                    save_data(path.split('.')[0] + '_' + str(IDs), new_image[0], new_mask[0])
                    IDs += 1
                    save_data(path.split('.')[0] + '_' + str(IDs), new_image[1], new_mask[1])
                    IDs += 1
                else:
                    save_data(path.split('.')[0] + '_' + str(IDs), new_image, new_mask)
                    IDs += 1


file_list = ['/home/stonebegin/Workspace/datasets/promise12/TrainingData_Part1/', '/home/stonebegin/Workspace/datasets/promise12/TrainingData_Part2/', '/home/stonebegin/Workspace/datasets/promise12/TrainingData_Part3/']
process_imgs(file_list)