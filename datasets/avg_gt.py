import glob
import h5py
import numpy as np

def load_data(path):
    with h5py.File(path, 'r') as input_file:
        raws = input_file['raw'][()]
        labels = input_file['label'][()]
        
    raws = np.array(raws)
    labels = np.array(labels)
    return labels

def save_data(filename, avg_gt):
    f = h5py.File('../data/' + filename + '.h5', 'w')
    f['avg_gt'] = avg_gt
    f.close()

root_path = '/home/stonebegin/Workspace/deep_learning/promise-3dunet/data/train_data/'
sum_labels = np.zeros([1, 16, 256, 256])
data_dir_list = glob.glob(root_path + '*')
for file_path in data_dir_list:
    labels = load_data(file_path)
    seg_data = labels
    seg_data = seg_data.reshape(1, 16, 256, 256)
    sum_labels += seg_data

mean_labels = sum_labels / len(data_dir_list)
save_data('avg_gt', mean_labels)