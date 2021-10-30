from .datasets.transforms import (RandomCrop, RandomMirror, RandomOverturn,
                                  RandomRotate, FourLRotate, RandomRescaleCrop,
                                  ToTensor, RealNoiseAdd, Rescale, Compose)
from .datasets.tasks_dict import tasks_dict
import numpy as np
import torch
import json
import os


def json_loader(dict_file_dir):
    with open(dict_file_dir, 'r') as data_file:
        return json.load(data_file)


syn_denoise_aug = {
    '1': lambda crop_size : Compose(
        [RealNoiseAdd(), RandomRescaleCrop(crop_size), FourLRotate(), RandomMirror(), ToTensor(), Rescale()]),
    '2': lambda crop_size : Compose(
        [RealNoiseAdd(), RandomCrop(crop_size), FourLRotate(), RandomMirror(), ToTensor(), Rescale()]),
    '3': lambda crop_size : Compose(
        [RealNoiseAdd(), RandomCrop(crop_size), RandomMirror(), ToTensor(), Rescale()]),
}


real_denoise_aug = {
    '1': lambda crop_size : Compose(
        [RandomRescaleCrop(crop_size), FourLRotate(), RandomMirror(), ToTensor(), Rescale()]),
    '2': lambda crop_size : Compose(
        [RandomCrop(crop_size), FourLRotate(), RandomMirror(), ToTensor(), Rescale]),
    '3': lambda crop_size : Compose(
        [RandomCrop(crop_size), RandomMirror(), ToTensor(), Rescale()]),
}

def build_joint_transforms(crop_size=None, tag='train', mode='syn', aug=1):
    aug = str(aug)
    if mode == 'syn':
        if tag == 'train':
            return syn_denoise_aug[aug](crop_size)
        elif tag == 'test':
            return Compose([
                ToTensor(),
                Rescale()
            ])
    elif mode == 'real':
        if tag == 'train':
            return real_denoise_aug[aug](crop_size)
        elif tag == 'test':
            return Compose([
                ToTensor(),
                Rescale(),
            ])


def build_dataset(dataset, cfg):
    data_root = cfg.DATASET.DATA_ROOT
    data_name = dataset
    task = cfg.DATASET.TASK

    if cfg.SEARCH.SEARCH_ON:
        crop_size = cfg.DATASET.CROP_SIZE
    else:
        crop_size = cfg.INPUT.CROP_SIZE_TRAIN

    data_list_dir = cfg.DATALOADER.DATA_LIST_DIR
    num_workers = cfg.DATALOADER.NUM_WORKERS

    batch_size = cfg.DATALOADER.BATCH_SIZE_TRAIN

    search_on = cfg.SEARCH.SEARCH_ON

    if dataset == 'CBD_syn':
        task_s='denoise'
        transform = build_joint_transforms(crop_size, tag='train', mode='syn', aug=cfg.DATALOADER.DATA_AUG)
    elif dataset == 'CBD_real':
        task_s = 'denoise_CBD_real'
        transform = build_joint_transforms(crop_size, tag='train', mode='real', aug=cfg.DATALOADER.DATA_AUG)
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
                                     cfg.DATASET.LOAD_ALL, False)
        dataset_a = tasks_dict[task_s](os.path.join(data_root, task), a_data_list, transform,
                                     cfg.DATASET.LOAD_ALL, False)

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


def build_joint_dataset(cfg):
    data_loader_list = []
    val_set_list = []
    for dataset in cfg.DATASET.TRAIN_DATASETS:
        data_loader, val_data_list = build_dataset(dataset, cfg)
        data_loader_list.append(data_loader)
        val_set_list.append(val_data_list)

    return data_loader_list, val_set_list

