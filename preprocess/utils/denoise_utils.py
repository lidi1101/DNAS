from glob import glob
from PIL import Image
import json
import os


def denoise_dict_build(args):
    dict_list = []
    for dataset in args.datasets:
        if dataset in ['BSD500_300', 'BSD500_200', 'Urben100', 'set14']:
            dict = []
            data_dir = os.path.join(args.data_root, args.task, dataset)

            im_list = glob(os.path.join(data_dir, '*.jpg'))
            if len(im_list) == 0:
                im_list = glob(os.path.join(data_dir, '*.png'))
            if len(im_list) == 0:
                im_list = glob(os.path.join(data_dir, '*.bmp'))

            im_list.sort()

            for im_dir in im_list:
                with Image.open(im_dir) as img:
                    w, h = img.width, img.height

                sample_info = {
                    'path': '/'.join(im_dir.split('/')[-2:]),
                    'width': int(w),
                    'height': int(h)
                }
                dict.append(sample_info)
            dict_list.append(dict)

        elif 'SIM_noise' in dataset:
            dict_train = []
            dict_test = []
            data_dir = os.path.join(args.data_root, args.task, dataset)

            # build dict file for training set
            train_clean_list = glob(os.path.join(data_dir, 'train/clean_img', '*.jpg'))
            train_noise_list = glob(os.path.join(data_dir, 'train/noise_img', '*.jpg'))
            train_clean_list.sort()
            train_noise_list.sort()

            for clean_dir, noise_dir in zip(train_clean_list, train_noise_list):
                clean_id = clean_dir.split('/')[-1][:-4]
                noise_id = noise_dir.split('/')[-1][:-4]
                assert clean_id == noise_id
                with Image.open(clean_dir) as img:
                    w, h = img.width, img.height

                sample_info = {
                    'path_clean': '/'.join(clean_dir.split('/')[-4:]),
                    'path_noise': '/'.join(noise_dir.split('/')[-4:]),
                    'width': int(w),
                    'height': int(h)
                }
                dict_train.append(sample_info)

            # build dict file for test set
            test_clean_list = glob(os.path.join(data_dir, 'test/clean_img', '*.jpg'))
            test_noise_list = glob(os.path.join(data_dir, 'test/noise_img', '*.jpg'))
            test_clean_list.sort()
            test_noise_list.sort()

            for clean_dir, noise_dir in zip(test_clean_list, test_noise_list):
                clean_id = clean_dir.split('/')[-1][:-4]
                noise_id = noise_dir.split('/')[-1][:-4]
                assert clean_id == noise_id
                with Image.open(clean_dir) as img:
                    w, h = img.width, img.height

                sample_info = {
                    'path_clean': '/'.join(clean_dir.split('/')[-4:]),
                    'path_noise': '/'.join(noise_dir.split('/')[-4:]),
                    'width': int(w),
                    'height': int(h)
                }
                dict_test.append(sample_info)

            dict_list.append([dict_train, dict_test])

        elif dataset == 'CBD_syn':
            dict = []
            data_dir = os.path.join(args.data_root, args.task, 'CBDNet_dataset/synthetic')

            im_list = glob(os.path.join(data_dir, '*.jpg'))
            if len(im_list) == 0:
                im_list = glob(os.path.join(data_dir, '*.png'))
            if len(im_list) == 0:
                im_list = glob(os.path.join(data_dir, '*.bmp'))

            im_list.sort()
            for im_dir in im_list:
                with Image.open(im_dir) as img:
                    w, h = img.width, img.height

                sample_info = {
                    'path': '/'.join(im_dir.split('/')[-3:]),
                    'width': int(w),
                    'height': int(h)
                }
                dict.append(sample_info)
            dict_list.append(dict)

        elif dataset == 'CBD_test':
            dict = []
            data_dir = os.path.join(args.data_root, args.task, 'CBDNet_dataset/test')

            im_list = glob(os.path.join(data_dir, '*.jpg'))
            if len(im_list) == 0:
                im_list = glob(os.path.join(data_dir, '*.png'))
            if len(im_list) == 0:
                im_list = glob(os.path.join(data_dir, '*.bmp'))

            im_list.sort()
            for im_dir in im_list:
                with Image.open(im_dir) as img:
                    w, h = img.width, img.height

                sample_info = {
                    'path': '/'.join(im_dir.split('/')[-3:]),
                    'width': int(w),
                    'height': int(h)
                }
                dict.append(sample_info)
            dict_list.append(dict)

        elif dataset == 'CBD_real':
            dict = []
            data_dir = os.path.join(args.data_root, args.task, 'CBDNet_dataset/real')
            # build dict file for training set
            sample_list = glob(os.path.join(data_dir, 'Batch_*'))
            sample_list.sort()
            for sample in sample_list:
                clean_img = glob(os.path.join(sample, '*Reference.bmp'))[0]
                noisy_imgs = glob(os.path.join(sample, '*Noisy.bmp'))

                with Image.open(clean_img) as img:
                    w, h = img.width, img.height

                sample_info = {
                    'path_clean': '/'.join(clean_img.split('/')[-4:]),
                    'width': int(w),
                    'height': int(h)
                }

                noisy_num = len(noisy_imgs)
                if noisy_num == 0: print('has no noisy img')

                for i in range(noisy_num):
                    sample_info['path_noise{}'.format(i)]='/'.join(noisy_imgs[i].split('/')[-4:])
                sample_info['noisy_num'] = noisy_num
                dict.append(sample_info)
            dict_list.append(dict)

    return dict_list

