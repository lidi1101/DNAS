import os
import json
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def json_loader(dict_file_dir):
    with open(dict_file_dir, 'r') as data_file:
        return json.load(data_file)


# This class is build for loading different datasets in denoise tasks
class Denoise_datasets(Dataset):
    def __init__(self, data_root, data_dict, transform, load_all=False, to_gray=False):
        self.data_root = data_root
        self.transform = transform
        self.load_all = True
        self.to_gray = to_gray
        if self.load_all is False:
            self.data_dict = data_dict
        else:
            self.data_dict = []
            for sample_info in data_dict:
                sample_data = Image.open(os.path.join(self.data_root, sample_info['path'])).copy()
                if sample_data.mode in ['RGBA']:
                    sample_data = sample_data.convert('RGB')
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
            if sample_data.mode in ['RGBA']:
                sample_data = sample_data.convert('RGB')
        else:
            sample_data = sample_info['data']

        if self.to_gray:
            sample_data = sample_data.convert('L')

        # crop (w_start, h_start, w_end, h_end)
        image = sample_data
        target = sample_data

        sample = {'image': image, 'target': target}
        sample = self.transform(sample)
        return sample['image'], sample['target']


# This class is build for loading the denoise dataset SIM_noise1800
class Denoise_SIM_noise1800(Dataset):
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
                clean_img = Image.open(os.path.join(self.data_root, sample_info['path_clean'])).copy()
                noise_img = Image.open(os.path.join(self.data_root, sample_info['path_noise'])).copy()
                width = sample_info['width']
                height = sample_info['height']
                sample = {
                    'clean_img': clean_img,
                    'noise_img': noise_img,
                    'width': width,
                    'height': height
                }
                self.data_dict.append(sample)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        sample_info = self.data_dict[idx]
        if self.load_all is False:
            clean_img = Image.open(os.path.join(self.data_root, sample_info['path_clean']))
            noise_img = Image.open(os.path.join(self.data_root, sample_info['path_noise']))
        else:
            clean_img = sample_info['clean_img']
            noise_img = sample_info['noise_img']

        if self.to_gray:
            clean_img, noise_img = clean_img.convert('L'), noise_img.convert('L')

        # crop (w_start, h_start, w_end, h_end)
        image = noise_img
        target = clean_img

        sample = {'image': image, 'target': target}
        sample = self.transform(sample)

        return sample['image'], sample['target']


class Denoise_CBD_real(Dataset):
    def __init__(self, source_dir, data_dict, transform, load_all=False, to_gray=False):
        self.source_dir = source_dir
        self.transform = transform
        self.load_all = load_all
        self.to_gray = to_gray
        if self.load_all is False:
            self.data_dict = data_dict
        else:
            self.data_dict = []
            for sample_info in data_dict:
                noisy_num = sample_info['noisy_num']
                clean_img = Image.open(os.path.join(self.source_dir, sample_info['path_clean'])).copy()
                width = sample_info['width']
                height = sample_info['height']
                sample = {
                    'clean_img': clean_img,
                    'width': width,
                    'height': height,
                    'noisy_num': noisy_num
                }
                for noisy_i in range(noisy_num):
                    sample['noisy_img{}'.format(noisy_i)] = \
                        Image.open(os.path.join(self.source_dir, sample_info['path_noise{}'.format(noisy_i)])).copy()
                self.data_dict.append(sample)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        sample_info = self.data_dict[idx]
        noisy_id = random.randint(0, sample_info['noisy_num'] - 1)
        if self.load_all is False:
            clean_img = Image.open(os.path.join(self.source_dir, sample_info['path_clean']))
            noise_img = Image.open(os.path.join(self.source_dir, sample_info['path_noise{}'.format(noisy_id)]))
        else:
            clean_img = sample_info['clean_img']
            noise_img = sample_info['noisy_img{}'.format(noisy_id)]

        if self.to_gray:
            clean_img, noise_img = clean_img.convert('L'), noise_img.convert('L')

        # crop (w_start, h_start, w_end, h_end)
        image = noise_img
        target = clean_img

        sample = {'image': image, 'target': target}
        sample = self.transform(sample)

        return sample['image'], sample['target']



