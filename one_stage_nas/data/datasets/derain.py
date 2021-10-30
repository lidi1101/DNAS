import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def json_loader(dict_file_dir):
    with open(dict_file_dir, 'r') as data_file:
        return json.load(data_file)


# This class is build for loading different datasets in derain tasks
class Derain_datasets(Dataset):
    def __init__(self, data_root, data_dict, transform, load_all=False, to_gray=False):
        self.data_root = data_root
        self.transform = transform
        self.load_all = load_all
        self.to_gray = to_gray
        if load_all is False:
            self.data_dict = data_dict
        else:
            self.data_dict = []
            for sample_info in data_dict:
                sample_data = Image.open(os.path.join(self.data_root, sample_info['path'])).copy()
                width = sample_info['width']
                height = sample_info['height']
                sample = {
                    'data': sample_data,
                    'width': width,
                    'height': height
                }
                self.data_dict.append(sample)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        sample_info = self.data_dict[idx]
        if self.load_all is False:
            sample_data = Image.open(os.path.join(self.data_root, sample_info['path']))
        else:
            sample_data = sample_info['data']

        if self.to_gray:
            sample_data = sample_data.convert('LA')

        width = sample_info['width']
        height = sample_info['height']

        # crop (w_start, h_start, w_end, h_end)
        image = sample_data.crop((width, 0, width * 2, height))
        target = sample_data.crop((0, 0, width, height))

        sample = {'image': image, 'target': target}
        sample = self.transform(sample)

        return sample['image'], sample['target']



