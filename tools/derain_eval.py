"""
Searching script
"""

import argparse
import os
import json
import torch
import sys
import numpy as np
sys.path.append('..')
from one_stage_nas.config import cfg
from one_stage_nas.data import build_transforms
from one_stage_nas.utils.misc import mkdir
from one_stage_nas.modeling.architectures import build_model
from PIL import Image
from one_stage_nas.utils.evaluation_metrics import SSIM, PSNR
import time


def crop(crop_size, w, h):
    # slide_step = crop_size - crop_size // 4
    slide_step = crop_size
    x1 = list(range(0, w-crop_size, slide_step))
    x1.append(w-crop_size)
    y1 = list(range(0, h-crop_size, slide_step))
    y1.append(h-crop_size)

    x2 = [x+crop_size for x in x1]
    y2 = [y+crop_size for y in y1]

    return x1, x2, y1, y2


def json_loader(dict_file_dir):
    with open(dict_file_dir, 'r') as data_file:
        return json.load(data_file)


def joint_patches(output_buffer, w, h):
    count_matrix = np.zeros((int(h), int(w), 3), dtype=np.float32)
    im_result = torch.from_numpy(np.zeros((int(h), int(w), 3), dtype=np.float32))
    gt_result = torch.from_numpy(np.zeros((int(h), int(w), 3), dtype=np.float32))

    for item in output_buffer:
        im_patch = item['im_patch']
        gt_patch = item['gt_patch']
        crop_position = item['crop_position']
        w0, w1, h0, h1 = int(crop_position[0]), int(crop_position[1]), int(crop_position[2]), int(crop_position[3])

        im_result[h0:h1, w0:w1] = im_result[h0:h1, w0:w1] + im_patch.transpose(0, 2).transpose(0, 1).contiguous()
        gt_result[h0:h1, w0:w1] = gt_result[h0:h1, w0:w1] + gt_patch.transpose(0, 2).transpose(0, 1).contiguous()
        count_matrix[h0:h1, w0:w1] = count_matrix[h0:h1, w0:w1] + 1.0

    return im_result / torch.from_numpy(count_matrix), gt_result / torch.from_numpy(count_matrix)


def evaluation(cfg):
    print('load test set')
    data_name = cfg.DATASET.DATA_NAME
    data_list_dir = cfg.DATALOADER.DATA_LIST_DIR
    data_dict = json_loader(os.path.join(data_list_dir, cfg.DATASET.TASK, data_name, 'test.json'))

    crop_size = cfg.DATASET.CROP_SIZE
    data_root = cfg.DATASET.DATA_ROOT

    test_dict = []
    for im_info in data_dict:

        w, h = im_info['width'], im_info['height']
        im_id = im_info['path'].split('/')[-1]

        assert w >= crop_size and h >= crop_size
        x1, x2, y1, y2 = crop(crop_size, int(w), int(h))

        for x_start, x_end in zip(x1, x2):
            for y_start, y_end in zip(y1, y2):

                sample_info = {
                    'path': os.path.join(data_root, cfg.DATASET.TASK, '/'.join(im_info['path'].split('/')[-3:])),
                    'im_id': im_id,
                    'width': w,
                    'height': h,
                    'x1': x_start,
                    'x2': x_end,
                    'y1': y_start,
                    'y2': y_end
                }
                test_dict.append(sample_info)


    print('model build')

    trained_model_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                  '{}_{}'.format(cfg.DATASET.DATA_NAME, cfg.RESULT_DIR),
                                  'train', 'models/model_best.pth'))

    if not os.path.exists(trained_model_dir):
        print(trained_model_dir)
        print('trained_model does not exist')
        return

    model = build_model(cfg)
    model = torch.nn.DataParallel(model).cuda()

    model_state_dict = torch.load(trained_model_dir).pop("model")
    try:
        model.load_state_dict(model_state_dict)
    except:
        model.module.load_state_dict(model_state_dict)

    print('evaluation')

    transforms = build_transforms(tag='test')


    model.eval()
    metric_SSIM = SSIM(window_size=11, channel=3, is_cuda=True)
    metric_PSNR = PSNR()

    batch_size = cfg.DATALOADER.BATCH_SIZE_TEST

    result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                '{}_{}'.format(cfg.DATASET.DATA_NAME, cfg.RESULT_DIR),
                                'eval', cfg.DATASET.DATA_NAME))

    mkdir(result_save_dir)

    with torch.no_grad():
        previous_im_id = ''
        current_im_id = ''
        previous_im_w = None
        previous_im_h = None
        output_buffer = []

        dict_len = len(test_dict)
        batch_index_end=0

        i = 0

        time_sum=0
        while batch_index_end < dict_len:

            batch_index_start = batch_index_end
            batch_index_end = min(batch_index_end + batch_size, dict_len)

            images = []
            targets = []
            im_id = []
            w, h = [], []
            x1, x2, y1, y2 = [], [], [], []

            for index in range(batch_index_start, batch_index_end):
                patch_info = test_dict[index]
                if patch_info['im_id'] != current_im_id:
                    sample_data = Image.open(patch_info['path'])
                    width = patch_info['width']
                    height = patch_info['height']
                    current_im_id = patch_info['im_id']

                p_x1, p_x2, p_y1, p_y2 = patch_info['x1'], patch_info['x2'], patch_info['y1'], patch_info['y2']

                image = sample_data.crop((width + p_x1, p_y1, width + p_x2, p_y2))
                target = sample_data.crop((p_x1, p_y1, p_x2, p_y2))

                sample = {'image': image, 'target': target}
                sample = transforms(sample)

                images.append(sample['image'])
                targets.append(sample['target'])
                im_id.append(patch_info['im_id'])
                w.append(width)
                h.append(height)
                x1.append(p_x1)
                x2.append(p_x2)
                y1.append(p_y1)
                y2.append(p_y2)

            images = torch.stack(images)
            targets = torch.stack(targets)
            start_time=time.time()
            output = model(images)
            torch.cuda.synchronize()
            end_time=time.time()
            time_sum+=(end_time-start_time)

            for j in range(images.size(0)):
                if not (i == 0 and j == 0) and im_id[j] != previous_im_id:
                    im_result, gt_result = joint_patches(output_buffer, previous_im_w, previous_im_h)
                    im_result[im_result > 1.0] = 1.0
                    im_result[im_result < 0.0] = 0.0

                    metric_SSIM(im_result.cuda(), gt_result.cuda())
                    metric_PSNR(im_result, gt_result)
                    im_PIL = Image.fromarray(np.array(im_result * 255, np.uint8))
                    im_PIL.save(result_save_dir + previous_im_id)
                    print('image: {} done'.format(previous_im_id))
                    output_buffer = []

                previous_im_id = im_id[j]
                previous_im_w = w[j]
                previous_im_h = h[j]

                patch_info = {
                    'im_patch': output[j].cpu(),
                    'gt_patch': targets[j],
                    'crop_position': [x1[j], x2[j], y1[j], y2[j]]
                }
                output_buffer.append(patch_info)

            i+=1

        im_result, gt_result = joint_patches(output_buffer, previous_im_w, previous_im_h)
        im_result[im_result>1.0] = 1.0
        im_result[im_result<0.0] = 0.0
        metric_SSIM(im_result.cuda(), gt_result.cuda())
        metric_PSNR(im_result, gt_result)
        im_PIL = Image.fromarray(np.array(im_result * 255, np.uint8))
        im_PIL.save(result_save_dir + previous_im_id)
        print('image: {} done'.format(previous_im_id))

    ssim = metric_SSIM.metric_get()
    psnr = metric_PSNR.metric_get()

    print('time cost:{}'.format(time_sum))

    print('dataset:{} ssim:{}, psnr:{}'.format(cfg.DATASET.DATA_NAME, ssim, psnr))
    with open((result_save_dir + '_evaluation_result.txt'), 'w') as f:
        f.write('SSIM:{} PSNR:{}'.format(ssim, psnr))


def main():
    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument(
        "--config-file",
        default="../configs/derain/amt_w_1/derain_inference.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    evaluation(cfg)

if __name__ == "__main__":
    main()
