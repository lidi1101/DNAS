from .datasets.transforms import (RandomCrop, RandomMirror, RandomOverturn,
                                  RandomRotate, FourLRotate, Normalize,
                                  RandomRescaleCrop, ToTensor, NoiseToTensor,
                                  Rescale, Compose)
from .datasets.tasks_dict import tasks_dict
import numpy as np
import torch
import json
import os


def json_loader(dict_file_dir):
    with open(dict_file_dir, 'r') as data_file:
        return json.load(data_file)


sim_denoise_aug = {
    '1': lambda crop_size : Compose([RandomCrop(crop_size), RandomRotate(), RandomMirror(), ToTensor(), Rescale()]),
    '2': lambda crop_size : Compose([RandomCrop(crop_size), FourLRotate(), RandomMirror(), ToTensor(), Rescale()]),
    '3': lambda crop_size: Compose(
        [RandomCrop(crop_size), RandomRotate(), RandomMirror(), RandomOverturn(), ToTensor(), Rescale()]),
    '4': lambda crop_size: Compose(
        [RandomCrop(crop_size), FourLRotate(), RandomMirror(), RandomOverturn(), ToTensor(), Rescale()]),
    '5': lambda crop_size: Compose(
        [RandomRescaleCrop(crop_size), FourLRotate(), RandomMirror(), ToTensor(), Rescale()]),

}

denoise_aug = {
    '1': lambda crop_size, sigma: Compose(
        [RandomCrop(crop_size), RandomRotate(), RandomMirror(), NoiseToTensor(sigma), Rescale()]),
    '2': lambda crop_size, sigma: Compose(
        [RandomCrop(crop_size), FourLRotate(), RandomMirror(), NoiseToTensor(sigma), Rescale()]),
    '3': lambda crop_size, sigma: Compose(
        [RandomCrop(crop_size), RandomRotate(), RandomMirror(), RandomOverturn(), NoiseToTensor(sigma), Rescale()]),
    '4': lambda crop_size, sigma: Compose(
        [RandomCrop(crop_size), FourLRotate(), RandomMirror(), RandomOverturn(), NoiseToTensor(sigma), Rescale()]),
    '5': lambda crop_size, sigma: Compose(
        [RandomRescaleCrop(crop_size), FourLRotate(), RandomMirror(), NoiseToTensor(sigma), Rescale()]),
}


def build_transforms(crop_size=None, task='derain', tag='train', sigma=[], aug=1):
    aug = str(aug)
    if task == 'derain':
        if tag == 'train':
            return Compose([
                RandomRescaleCrop(crop_size),
                RandomMirror(),
                ToTensor(),
                Rescale(),
            ])
        elif tag == 'test':
            return Compose([
                ToTensor(),
                Rescale(),
            ])
    elif task == 'denoise':
        if sigma=='SIM':
            if tag == 'train':
                return sim_denoise_aug[aug](crop_size)
            elif tag == 'test':
                return Compose([
                    ToTensor(),
                    Rescale(),
                ])
        else:
            if tag == 'train':
                return denoise_aug[aug](crop_size, sigma)
            elif tag == 'test':
                return Compose([
                    NoiseToTensor(sigma),
                    Rescale(),
                ])


def build_dataset(cfg):
    data_root = cfg.DATASET.DATA_ROOT#/home/lidi/Documents/BSR/BSD500/data
    data_name = cfg.DATASET.DATA_NAME#BSD500_300

    task = cfg.DATASET.TASK
    if task == 'denoise' and 'SIM_noise1800' in data_name:
        task_s = 'denoise_SIM_noise1800'
    else:
        task_s = task

    if cfg.SEARCH.SEARCH_ON:
        crop_size = cfg.DATASET.CROP_SIZE
    else:
        crop_size = cfg.INPUT.CROP_SIZE_TRAIN

    data_list_dir = cfg.DATALOADER.DATA_LIST_DIR
    num_workers = cfg.DATALOADER.NUM_WORKERS

    batch_size = cfg.DATALOADER.BATCH_SIZE_TRAIN

    search_on = cfg.SEARCH.SEARCH_ON

    if 'SIM_noise1800' in data_name:
        transform = build_transforms(crop_size, task, tag='train', sigma='SIM', aug=cfg.DATALOADER.DATA_AUG)
    else:
        transform = build_transforms(crop_size, task, tag='train', sigma=cfg.DATALOADER.SIGMA, aug=cfg.DATALOADER.DATA_AUG)
    if task in ['derain']:
        data_dict = json_loader(os.path.join(data_list_dir, task, data_name, 'train.json'))
    elif task in ['denoise']:
        data_dict = json_loader(os.path.join(data_list_dir, task, data_name + '.json'))

    if search_on:
        num_samples = len(data_dict)
        val_split = int(np.floor(cfg.SEARCH.VAL_PORTION * num_samples))
        num_train = num_samples - val_split
        train_split = int(np.floor(cfg.SEARCH.PORTION * num_train))
        w_data_list = [data_dict[i] for i in range(train_split)]
        a_data_list = [data_dict[i] for i in range(train_split, num_train)]
        v_data_list = [data_dict[i] for i in range(num_train, num_samples)]

        dataset_w = tasks_dict[task_s](os.path.join(data_root, task), w_data_list, transform,
                                     cfg.DATASET.LOAD_ALL, cfg.DATASET.TO_GRAY)
        dataset_a = tasks_dict[task_s](os.path.join(data_root, task), a_data_list, transform,
                                     cfg.DATASET.LOAD_ALL, cfg.DATASET.TO_GRAY)

        data_loader_w = torch.utils.data.DataLoader(
            dataset_w,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True)

        data_loader_a = torch.utils.data.DataLoader(
            dataset_a,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True)

        return [data_loader_w, data_loader_a], v_data_list
    else:
        num_samples = len(data_dict)
        val_split = int(np.floor(cfg.SEARCH.VAL_PORTION * num_samples))
        num_train = num_samples - val_split

        t_data_list = [data_dict[i] for i in range(num_train)]
        v_data_list = [data_dict[i] for i in range(num_train, num_samples)]

        dataset_t = tasks_dict[task_s](os.path.join(data_root, task), t_data_list, transform,
                                     cfg.DATASET.LOAD_ALL, cfg.DATASET.TO_GRAY)

        data_loader_t = torch.utils.data.DataLoader(
            dataset_t,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True)

        return data_loader_t, v_data_list

