from utils import (dict_build, json_save, make_if_not_exist)
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='dataset preprocess')
    parser.add_argument('--data_root', type=str, default='C:\\Users\\LIDI\\Desktop\\RealNoiseHKPoly')
    parser.add_argument('--task', type=str, default='derain')
    parser.add_argument('--dataset', type=str, default='polyu')
    parser.add_argument('--json_dir', type=str, default='dataset_json')
    args = parser.parse_args()

    data_source_dir = os.path.join(args.data_root, args.task, args.dataset)
    train_dict, test_dict = dict_build(data_source_dir)

    json_save_dir = os.path.join(args.json_dir, args.task, args.dataset)
    make_if_not_exist(json_save_dir)
    #json_save(os.path.join(json_save_dir, 'train.json'), train_dict)
    json_save(os.path.join(json_save_dir, 'test.json'), test_dict)


if __name__ == '__main__':
    main()




