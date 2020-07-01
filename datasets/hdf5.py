import collections
import importlib
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from unet3d.utils import get_logger

logger = get_logger('HDF5Dataset')


class HDF5Dataset(Dataset):
    def __init__(self, file_path, phase):

        assert phase in ['train', 'val', 'test']
        print('Phase now: ', phase)
        # file_path: train_data path / val_data path / test_data path
        self.file_path = file_path  # datasets/train_data/
        self.data_dir_list = glob.glob(self.file_path + '*')


    def __getitem__(self, idx):
        raws, labels = self._loader(self.data_dir_list[idx])
        # raws = np.rollaxis(raws, 3, 1)
        # labels = np.rollaxis(labels, 3, 1)
        # raws = self.crop_img(raws)
        # labels = self.crop_img(labels)
        img_data = self._normalization(raws)
        seg_data = labels
        img_data = img_data.reshape(1, 16, 256, 256)
        seg_data = seg_data.reshape(1, 16, 256, 256)

        # label 1,2,4 和背景区域做二分类
        # print('Before: ', seg_data.shape)
        # seg_mask = np.zeros((1, 128, 160, 160))
        # seg_mask[0] = ( (seg_data[0] + seg_data[1] + seg_data[2]) > 0.1 ).astype(int)
        # print('After: ', seg_mask.shape)

        # label 1,4 和 2做二分类
        # print('Before: ', seg_data.shape)
        # seg_mask = np.zeros((2, 128, 160, 160))
        # seg_mask[0] = ( (seg_data[0] + seg_data[2]) > 0.1 ).astype(int)
        # seg_mask[1] = ( seg_data[1] > 0.1 ).astype(int)
        # seg_data = ( (seg_data[0] + seg_data[2]) > 0.1 ).astype(int)
        # print('After: ', seg_data.shape)

        # label 1 和 4 做二分类
        # print('Before: ', seg_data.shape)
        # seg_mask = np.zeros((2, 128, 160, 160))
        # seg_mask[0] = seg_data[0]
        # seg_mask[1] = seg_data[2]
        # seg_data = ( (seg_data[0] + seg_data[2]) > 0.1 ).astype(int)
        # print('After: ', seg_data.shape)


        return img_data, seg_data


    def __len__(self):
        return len(self.data_dir_list)


    @staticmethod
    def _loader(path):
        with h5py.File(path, 'r') as input_file:
            raws = input_file['raw'][()]
            labels = input_file['label'][()]
            
        raws = np.array(raws)
        labels = np.array(labels)
        return raws, labels 


    @staticmethod
    def _normalization(img_data):
        # 归一化
        img_nonzero = img_data[np.nonzero(img_data)]
        img = (img_data - np.mean(img_nonzero)) / np.std(img_nonzero)
        img[img == img.min()] = 0
        return img

    @staticmethod
    def crop_img(img_data):
        img_data = img_data[:, 13:141, 40:200, 40:200]  # shape: (4, 128, 160, 160)
        return img_data


def get_train_loaders(config):

    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']
    logger.info('Creating training and validation set loaders...')
    
    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.get('batch_size', 1)
    logger.info(f'Batch size for train/val loader: {batch_size}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints

    train_dataset = HDF5Dataset(loaders_config['train_path'], phase='train')
    val_dataset = HDF5Dataset(loaders_config['val_path'], phase='val')

    return {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    }

def get_test_loaders(config):

    assert 'datasets' in config, 'Could not find data sets configuration'
    datasets_config = config['datasets']

    test_path = datasets_config['test_path']

    num_workers = datasets_config.get('num_workers', 1)
    logger.info(f'Number of workers for the dataloader: {num_workers}')

    batch_size = datasets_config.get('batch_size', 1)
    logger.info(f'Batch size for dataloader: {batch_size}')

    # construct datasets lazily
    test_dataset = HDF5Dataset(test_path, phase='test')

    # img_data, seg_data
    return {'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)}