from preprocess.utils import (denoise_dict_build, json_save, make_if_not_exist)
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description='dataset preprocess')
    # parser.add_argument('--data_root', type=str, default='/home/hkzhang/Documents/sdb_a/nas_data')
    parser.add_argument('--data_root', type=str, default='/home/lidi/Documents/BSR/BSD500/data')
    parser.add_argument('--task', type=str, default='denoise')
    # parser.add_argument('--datasets', type=str, default=['BSD500_300', 'BSD500_200', 'Urben100', 'set14'])
   # parser.add_argument('--datasets', type=str, default=['CBD_real', 'CBD_test', 'CBD_syn'])
    parser.add_argument('--datasets', type=str, default=['BSD500_300', 'BSD500_200'])
    parser.add_argument('--json_dir', type=str, default='dataset_json')
    # parser.add_argument('--crop_size', type=int, default=64)
    args = parser.parse_args()


    dict_list = denoise_dict_build(args)

    json_save_dir = os.path.join(args.json_dir, args.task)
    make_if_not_exist(json_save_dir)
    for dataset, dict in zip(args.datasets, dict_list):
        # CBD_real and CBD_test datasets are RENOIR and DND dataset, respectively.
        if dataset in ['BSD500_300', 'BSD500_200', 'Urben100', 'set14', 'CBD_syn', 'CBD_real', 'CBD_test']:
            json_save(os.path.join(json_save_dir, '{}.json'.format(dataset)), dict)
            print(os.path.join(json_save_dir, '{}.json'.format(dataset)), dict)
        elif dataset in ['SIM_noise1800', 'SIM_noise_real']:
            json_save(os.path.join(json_save_dir, '{}_train.json'.format(dataset)), dict[0])
            json_save(os.path.join(json_save_dir, '{}_test.json'.format(dataset)), dict[1])


if __name__ == '__main__':
    main()




