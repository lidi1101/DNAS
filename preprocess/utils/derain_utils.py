from glob import glob
from PIL import Image
import json
import os


# def crop(crop_size, w, h):
#     slide_step = crop_size - crop_size // 4
#     # slide_step = crop_size
#     x1 = list(range(0, w-crop_size, slide_step))
#     x1.append(w-crop_size)
#     y1 = list(range(0, h-crop_size, slide_step))
#     y1.append(h-crop_size)
#
#     x2 = [x+crop_size for x in x1]
#     y2 = [y+crop_size for y in y1]
#
#     return x1, x2, y1, y2



def dict_build(data_dir):
    train_dict=[]
    test_dict=[]
    train_im_list = glob(os.path.join(data_dir, 'train', '*.jpg'))
    test_im_list = glob(os.path.join(data_dir, 'test', '*.jpg'))

    for im_dir in train_im_list:
        with Image.open(im_dir) as img:
            w, h = img.width/2, img.height

        # assert w >= crop_size and h >= crop_size
        sample_info={
            'path': '/'.join(im_dir.split('\\')[-3:]),
            'width': int(w),
            'height': int(h)
            }
        train_dict.append(sample_info)

    for im_dir in test_im_list:
        with Image.open(im_dir) as img:
            w, h = img.width / 2, img.height

        # assert w >= crop_size and h >= crop_size
        sample_info = {
            'path': '/'.join(im_dir.split('\\')[-3:]),
            'width': int(w),
            'height': int(h)
        }
        test_dict.append(sample_info)

    # for im_dir in test_im_list:
    #     with Image.open(im_dir) as img:
    #         w, h = img.width / 2, img.height
    #
    #     assert w >= crop_size and h >= crop_size
    #     x1, x2, y1, y2 = crop(crop_size, int(w), int(h))
    #
    #     im_id = im_dir.split('/')[-1]
    #     for x_start, x_end in zip(x1, x2):
    #         for y_start, y_end in zip(y1, y2):
    #
    #             sample_info = {
    #                 'path': '/'.join(im_dir.split('/')[-3:]),
    #                 'im_id': im_id,
    #                 'width': w,
    #                 'height': h,
    #                 'x1': x_start,
    #                 'x2': x_end,
    #                 'y1': y_start,
    #                 'y2': y_end
    #             }
    #             test_dict.append(sample_info)

    return train_dict, test_dict


def json_save(save_path, dict_file):
    with open(save_path, 'w') as f:
        json.dump(dict_file, f)


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)






