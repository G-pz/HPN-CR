import os
import numpy as np
import rasterio
import csv
import torch
from torch.utils.data import Dataset
from utils.feature_detectors import get_cloud_mask

class AlignedDataset(Dataset):

    def __init__(self, opts, filelist):
        self.opts = opts

        self.filelist = filelist
        self.n_images = len(self.filelist)

        self.clip_min = [[-25.0, -32.5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.clip_max = [[0, 0], [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
                    [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]]

        self.max_val = 1
        self.scale = 10000


    def __getitem__(self, index):

        fileID = self.filelist[index]

        s1_path = os.path.join(self.opts.input_data_folder, fileID[1], fileID[4])
        s2_cloudfree_path = os.path.join(self.opts.input_data_folder, fileID[2], fileID[4])
        s2_cloudy_path = os.path.join(self.opts.input_data_folder, fileID[3], fileID[4])
        s1_data = self.get_sar_image(s1_path).astype('float32')
        s2_cloudfree_data = self.get_opt_image(s2_cloudfree_path).astype('float32')
        s2_cloudy_data = self.get_opt_image(s2_cloudy_path).astype('float32')
        cloud_coverage = get_cloud_mask(s2_cloudy_data, cloud_threshold=0.2, binarize=True)

        s1_data = self.get_normalized_data(s1_data, data_type=1)
        s2_cloudfree_data = self.get_normalized_data(s2_cloudfree_data, data_type=2)
        s2_cloudy_data = self.get_normalized_data(s2_cloudy_data, data_type=3)


        s1_data = torch.from_numpy(s1_data)
        cloudy_data = torch.from_numpy(s2_cloudy_data)
        source_data = torch.concat((cloudy_data, s1_data), dim=0)
        results = {'cloudy_data': cloudy_data,
                   'target': s2_cloudfree_data,
                   'source': source_data,
                   's1_data': s1_data,
                   'file_name': fileID[4],
                   'cloud_coverage': cloud_coverage}

        return results

    def __len__(self):
        return self.n_images

    def get_opt_image(self, path):

        src = rasterio.open(path, 'r', driver='GTiff')
        image = src.read()
        src.close()
        image[np.isnan(image)] = np.nanmean(image)  

        return image

    def get_sar_image(self, path):

        src = rasterio.open(path, 'r', driver='GTiff')
        image = src.read()
        src.close()
        image[np.isnan(image)] = np.nanmean(image)  
        return image

    def get_normalized_data(self, data_image, data_type):
        # SAR
        if data_type == 1:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel], self.clip_max[data_type - 1][channel])
                data_image[channel] -= self.clip_min[data_type - 1][channel]
                data_image[channel] = self.max_val * (data_image[channel] / (self.clip_max[data_type - 1][channel] - self.clip_min[data_type - 1][channel]))
        # OPT
        elif data_type == 2 or data_type == 3:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel], self.clip_max[data_type - 1][channel])
            data_image /= self.scale

        return data_image
'''
read data.csv
'''
def get_train_val_test_filelists(listpath):

    csv_file = open(listpath, "r")
    list_reader = csv.reader(csv_file)

    train_filelist = []
    val_filelist = []
    test_filelist = []
    for f in list_reader:
        line_entries = f
        if line_entries[0] == '1':
            train_filelist.append(line_entries)
        elif line_entries[0] == '2':
            val_filelist.append(line_entries)
        elif line_entries[0] == '3':
            test_filelist.append(line_entries)

    csv_file.close()

    return train_filelist, val_filelist, test_filelist
