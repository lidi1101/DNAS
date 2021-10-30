import os
import logging
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from one_stage_nas.utils.evaluation_metrics import SSIM, PSNR
from one_stage_nas.data import build_joint_transforms, build_transforms, RealNoiseAdd
from one_stage_nas.utils.misc import mkdir


def joint_patches(output_buffer, w, h, channel=3):
    if channel==3:
        count_matrix = np.zeros((int(h), int(w), 3), dtype=np.float32)
        im_result = torch.from_numpy(np.zeros((int(h), int(w), 3), dtype=np.float32))
        gt_result = torch.from_numpy(np.zeros((int(h), int(w), 3), dtype=np.float32))
    elif channel==1:
        count_matrix = np.zeros((int(h), int(w), 1), dtype=np.float32)
        im_result = torch.from_numpy(np.zeros((int(h), int(w), 1), dtype=np.float32))
        gt_result = torch.from_numpy(np.zeros((int(h), int(w), 1), dtype=np.float32))

    for item in output_buffer:
        im_patch = item['im_patch']
        gt_patch = item['gt_patch']
        crop_position = item['crop_position']
        w0, w1, h0, h1 = int(crop_position[0]), int(crop_position[1]), int(crop_position[2]), int(crop_position[3])

        im_result[h0:h1, w0:w1] = im_result[h0:h1, w0:w1] + im_patch.transpose(0, 2).transpose(0, 1).contiguous()
        gt_result[h0:h1, w0:w1] = gt_result[h0:h1, w0:w1] + gt_patch.transpose(0, 2).transpose(0, 1).contiguous()
        count_matrix[h0:h1, w0:w1] = count_matrix[h0:h1, w0:w1] + 1.0
    return im_result / torch.from_numpy(count_matrix), gt_result / torch.from_numpy(count_matrix)


def crop(crop_size, w, h):
    slide_step = crop_size - crop_size // 4
    # slide_step = crop_size
    x1 = list(range(0, w-crop_size, slide_step))
    x1.append(w-crop_size)
    y1 = list(range(0, h-crop_size, slide_step))
    y1.append(h-crop_size)

    x2 = [x+crop_size for x in x1]
    y2 = [y+crop_size for y in y1]

    return x1, x2, y1, y2


def truncated(input_tensor, max_l=1.0, min_l=0.0):
    input_tensor[input_tensor>max_l] = max_l
    input_tensor[input_tensor<min_l] = min_l

    return input_tensor


def tensor2img(input, output, target):
    b, c, h, w = target.shape

    input_img = []
    output_img = []
    target_img = []

    for i in range(b):
        input_img.append(input[i])
        output_img.append(output[i])
        target_img.append(target[i])

    return torch.cat(input_img, 1), torch.cat(output_img, 1), torch.cat(target_img, 1)


def denoise_inference(model, test_list, cfg, show_img=False, tag='search'):
    logger = logging.getLogger("one_stage_nas.inference")
    print('load test set')

    crop_size = cfg.DATASET.CROP_SIZE
    data_root = cfg.DATASET.DATA_ROOT

    test_dict = []
    for im_info in test_list:

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

    print('evaluation')
    transforms = build_transforms(task='denoise', tag='test', sigma=cfg.DATALOADER.SIGMA)

    model.eval()
    metric_SSIM = SSIM(window_size=11, channel=cfg.MODEL.IN_CHANNEL, is_cuda=True)
    metric_PSNR = PSNR()

    batch_size = cfg.DATALOADER.BATCH_SIZE_TEST

    if tag == 'search':
        result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                    '{}_{}'.format(cfg.DATASET.DATA_NAME, cfg.RESULT_DIR), 'search/img_result'))
    elif tag == 'train':
        result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                    '{}_{}'.format(cfg.DATASET.DATA_NAME, cfg.RESULT_DIR),
                                    'train_noise_{}/img_result'.format(cfg.DATALOADER.SIGMA)))
    mkdir(result_save_dir)

    with torch.no_grad():
        previous_im_id = ''
        current_im_id = ''
        previous_im_w = None
        previous_im_h = None
        output_buffer = []

        dict_len = len(test_dict)
        batch_index_end = 0

        show_id = np.random.randint(0, dict_len // batch_size-1, 2)

        input_imgs = []
        output_imgs = []
        target_imgs = []

        i = 0
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

                image = sample_data.crop((p_x1, p_y1, p_x2, p_y2))
                if cfg.DATASET.TO_GRAY:
                    image = image.convert('L')
                target = image

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
            output = model(images)

            if show_img and i in show_id:
                input_img, output_img, target_img = tensor2img(images, output, targets)
                input_imgs.append(input_img)
                output_imgs.append(output_img.cpu())
                target_imgs.append(target_img)

            for j in range(images.size(0)):
                if not (i == 0 and j == 0) and im_id[j] != previous_im_id:
                    im_result, gt_result = joint_patches(output_buffer, previous_im_w, previous_im_h, cfg.MODEL.IN_CHANNEL)
                    im_result[im_result > 1.0] = 1.0
                    im_result[im_result < 0.0] = 0.0

                    metric_SSIM(im_result.cuda(), gt_result.cuda())
                    metric_PSNR(im_result, gt_result)
                    im_PIL = Image.fromarray(np.array(im_result.squeeze() * 255, np.uint8))
                    im_PIL.save(os.path.join(result_save_dir, previous_im_id))
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

            i += 1

        im_result, gt_result = joint_patches(output_buffer, previous_im_w, previous_im_h, cfg.MODEL.IN_CHANNEL)
        im_result[im_result > 1.0] = 1.0
        im_result[im_result < 0.0] = 0.0
        metric_SSIM(im_result.cuda(), gt_result.cuda())
        metric_PSNR(im_result, gt_result)
        im_PIL = Image.fromarray(np.array(im_result.squeeze() * 255, np.uint8))
        im_PIL.save(os.path.join(result_save_dir, previous_im_id))

    ssim = metric_SSIM.metric_get()
    psnr = metric_PSNR.metric_get()

    logger.info(' Val: SSIM:{} PSNR:{}'.format(ssim, psnr))

    if show_img:
        return ssim, psnr, truncated(torch.cat(input_imgs, 2), 1.0, 0.0), \
               truncated(torch.cat(output_imgs, 2), 1.0, 0.0), \
               torch.cat(target_imgs, 2)
    else:
        return ssim, psnr


def denoise_SIM_noise1800_inference(model, test_list, cfg, show_img=False, tag='search'):
    logger = logging.getLogger("one_stage_nas.inference")
    print('load test set')

    crop_size = cfg.DATASET.CROP_SIZE
    data_root = cfg.DATASET.DATA_ROOT

    test_dict = []
    for im_info in test_list:

        w, h = im_info['width'], im_info['height']
        im_id = im_info['path_clean'].split('/')[-1]

        assert w >= crop_size and h >= crop_size
        x1, x2, y1, y2 = crop(crop_size, int(w), int(h))

        for x_start, x_end in zip(x1, x2):
            for y_start, y_end in zip(y1, y2):
                sample_info = {
                    'path_clean': os.path.join(data_root, cfg.DATASET.TASK, '/'.join(im_info['path_clean'].split('/')[-4:])),
                    'path_noise': os.path.join(data_root, cfg.DATASET.TASK, '/'.join(im_info['path_noise'].split('/')[-4:])),
                    'im_id': im_id,
                    'width': w,
                    'height': h,
                    'x1': x_start,
                    'x2': x_end,
                    'y1': y_start,
                    'y2': y_end
                }
                test_dict.append(sample_info)

    print('evaluation')
    transforms = build_transforms(task='denoise', tag='test', sigma='SIM')

    model.eval()
    metric_SSIM = SSIM(window_size=11, channel=cfg.MODEL.IN_CHANNEL, is_cuda=True)
    metric_PSNR = PSNR()

    batch_size = cfg.DATALOADER.BATCH_SIZE_TEST

    if tag == 'search':
        result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                    '{}_{}'.format(cfg.DATASET.DATA_NAME, cfg.RESULT_DIR), 'search/img_result'))
    elif tag == 'train':
        result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                    '{}_{}'.format(cfg.DATASET.DATA_NAME, cfg.RESULT_DIR),
                                    'train/img_result'))
    mkdir(result_save_dir)

    with torch.no_grad():
        previous_im_id = ''
        current_im_id = ''
        previous_im_w = None
        previous_im_h = None
        output_buffer = []

        dict_len = len(test_dict)
        batch_index_end = 0

        show_id = np.random.randint(0, dict_len // batch_size-1, 2)

        input_imgs = []
        output_imgs = []
        target_imgs = []

        i = 0
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
                    clean_img = Image.open(patch_info['path_clean'])
                    noise_img = Image.open(patch_info['path_noise'])
                    width = patch_info['width']
                    height = patch_info['height']
                    current_im_id = patch_info['im_id']

                p_x1, p_x2, p_y1, p_y2 = patch_info['x1'], patch_info['x2'], patch_info['y1'], patch_info['y2']

                image = noise_img.crop((p_x1, p_y1, p_x2, p_y2))
                target = clean_img.crop((p_x1, p_y1, p_x2, p_y2))
                if cfg.DATASET.TO_GRAY:
                    image, target = image.convert('L'), target.convert('L')


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
            output = model(images)

            if show_img and i in show_id:
                input_img, output_img, target_img = tensor2img(images, output, targets)
                input_imgs.append(input_img)
                output_imgs.append(output_img.cpu())
                target_imgs.append(target_img)

            for j in range(images.size(0)):
                if not (i == 0 and j == 0) and im_id[j] != previous_im_id:
                    im_result, gt_result = joint_patches(output_buffer, previous_im_w, previous_im_h, cfg.MODEL.IN_CHANNEL)
                    im_result[im_result > 1.0] = 1.0
                    im_result[im_result < 0.0] = 0.0

                    metric_SSIM(im_result.cuda(), gt_result.cuda())
                    metric_PSNR(im_result, gt_result)
                    im_PIL = Image.fromarray(np.array(im_result.squeeze() * 255, np.uint8))
                    im_PIL.save(os.path.join(result_save_dir, previous_im_id))
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

            i += 1

        im_result, gt_result = joint_patches(output_buffer, previous_im_w, previous_im_h, cfg.MODEL.IN_CHANNEL)
        im_result[im_result > 1.0] = 1.0
        im_result[im_result < 0.0] = 0.0
        metric_SSIM(im_result.cuda(), gt_result.cuda())
        metric_PSNR(im_result, gt_result)
        im_PIL = Image.fromarray(np.array(im_result.squeeze() * 255, np.uint8))
        im_PIL.save(os.path.join(result_save_dir, previous_im_id))

    ssim = metric_SSIM.metric_get()
    psnr = metric_PSNR.metric_get()

    logger.info(' Val: SSIM:{} PSNR:{}'.format(ssim, psnr))

    if show_img:
        return ssim, psnr, truncated(torch.cat(input_imgs, 2), 1.0, 0.0), \
               truncated(torch.cat(output_imgs, 2), 1.0, 0.0), \
               torch.cat(target_imgs, 2)
    else:
        return ssim, psnr


def CBD_real_denoise_inference(model, test_list, cfg, show_img=False, tag='search'):
    print('load test set : CBD_real')

    crop_size = cfg.DATASET.CROP_SIZE
    data_root = cfg.DATASET.DATA_ROOT

    test_dict = []
    for im_info in test_list:
        # im_info['width'], im_info['height'] = 2400, 2400

        w, h = im_info['width'], im_info['height']
        im_id = im_info['path_clean'].split('/')[-1]

        assert w >= crop_size and h >= crop_size
        x1, x2, y1, y2 = crop(crop_size, int(w), int(h))

        for i in range(im_info['noisy_num']):
            im_id = im_id[:-4] + '_{}.bmp'.format(i)
            for x_start, x_end in zip(x1, x2):
                for y_start, y_end in zip(y1, y2):
                    sample_info = {
                        'path_clean': os.path.join(data_root, cfg.DATASET.TASK,
                                                   '/'.join(im_info['path_clean'].split('/')[-4:])),
                        'path_noise': os.path.join(data_root, cfg.DATASET.TASK,
                                                   '/'.join(im_info['path_noise{}'.format(i)].split('/')[-4:])),
                        'im_id': im_id,
                        'width': w,
                        'height': h,
                        'x1': x_start,
                        'x2': x_end,
                        'y1': y_start,
                        'y2': y_end
                    }
                    test_dict.append(sample_info)

    print('evaluation')
    transforms = build_joint_transforms(tag='test', mode='real')

    model.eval()
    metric_SSIM = SSIM(window_size=11, channel=cfg.MODEL.IN_CHANNEL, is_cuda=True)
    metric_PSNR = PSNR()

    batch_size = cfg.DATALOADER.BATCH_SIZE_TEST

    if tag == 'search':
        result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                    '{}_{}'.format('&'.join(cfg.DATASET.TRAIN_DATASETS), cfg.RESULT_DIR),
                                    'search/img_result_CBD_real'))
    elif tag == 'train':
        result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                    '{}_{}'.format('&'.join(cfg.DATASET.TRAIN_DATASETS), cfg.RESULT_DIR),
                                    'train_joint/img_result_CBD_real'))
    mkdir(result_save_dir)

    with torch.no_grad():
        previous_im_id = ''
        current_im_id = ''
        previous_im_w = None
        previous_im_h = None
        output_buffer = []

        dict_len = len(test_dict)
        batch_index_end = 0

        show_id = np.random.randint(0, dict_len // batch_size - 1, 2)

        input_imgs = []
        output_imgs = []
        target_imgs = []

        i = 0
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
                    clean_img = Image.open(patch_info['path_clean'])
                    noise_img = Image.open(patch_info['path_noise'])
                    width = patch_info['width']
                    height = patch_info['height']
                    current_im_id = patch_info['im_id']

                p_x1, p_x2, p_y1, p_y2 = patch_info['x1'], patch_info['x2'], patch_info['y1'], patch_info['y2']

                image = noise_img.crop((p_x1, p_y1, p_x2, p_y2))
                target = clean_img.crop((p_x1, p_y1, p_x2, p_y2))
                if cfg.DATASET.TO_GRAY:
                    image, target = image.convert('L'), target.convert('L')

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
            output = model(images)

            if show_img and i in show_id:
                input_img, output_img, target_img = tensor2img(images, output, targets)
                input_imgs.append(input_img)
                output_imgs.append(output_img.cpu())
                target_imgs.append(target_img)

            for j in range(images.size(0)):
                if not (i == 0 and j == 0) and im_id[j] != previous_im_id:
                    im_result, gt_result = joint_patches(output_buffer, previous_im_w, previous_im_h,
                                                         cfg.MODEL.IN_CHANNEL)
                    im_result[im_result > 1.0] = 1.0
                    im_result[im_result < 0.0] = 0.0

                    metric_SSIM(im_result.cuda(), gt_result.cuda())
                    metric_PSNR(im_result, gt_result)
                    im_PIL = Image.fromarray(np.array(im_result.squeeze() * 255, np.uint8))
                    im_PIL.save(os.path.join(result_save_dir, previous_im_id))
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

            i += 1

        im_result, gt_result = joint_patches(output_buffer, previous_im_w, previous_im_h, cfg.MODEL.IN_CHANNEL)
        im_result[im_result > 1.0] = 1.0
        im_result[im_result < 0.0] = 0.0
        metric_SSIM(im_result.cuda(), gt_result.cuda())
        metric_PSNR(im_result, gt_result)
        im_PIL = Image.fromarray(np.array(im_result.squeeze() * 255, np.uint8))
        im_PIL.save(os.path.join(result_save_dir, previous_im_id))

    ssim = metric_SSIM.metric_get()
    psnr = metric_PSNR.metric_get()


    if show_img:
        return ssim, psnr, metric_SSIM.im_count, truncated(torch.cat(input_imgs, 2), 1.0, 0.0), \
               truncated(torch.cat(output_imgs, 2), 1.0, 0.0), \
               torch.cat(target_imgs, 2)
    else:
        return ssim, psnr, metric_SSIM.im_count


def CBD_syn_denoise_inference(model, test_list, cfg, show_img=False, tag='search'):
    print('load test set : CBD_syn')

    crop_size = cfg.DATASET.CROP_SIZE
    data_root = cfg.DATASET.DATA_ROOT

    test_dict = []
    for im_info in test_list:

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

    print('evaluation')
    transforms = build_joint_transforms(tag='test', mode='syn')

    model.eval()
    metric_SSIM = SSIM(window_size=11, channel=cfg.MODEL.IN_CHANNEL, is_cuda=True)
    metric_PSNR = PSNR()

    batch_size = cfg.DATALOADER.BATCH_SIZE_TEST

    if tag == 'search':
        result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                    '{}_{}'.format('&'.join(cfg.DATASET.TRAIN_DATASETS), cfg.RESULT_DIR),
                                    'search/img_result_CBD_syn'))
    elif tag == 'train':
        result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                    '{}_{}'.format('&'.join(cfg.DATASET.TRAIN_DATASETS), cfg.RESULT_DIR),
                                    'train_joint/img_result_CBD_syn'))
    mkdir(result_save_dir)

    AddNoise = RealNoiseAdd()

    with torch.no_grad():
        previous_im_id = ''
        current_im_id = ''
        previous_im_w = None
        previous_im_h = None
        output_buffer = []

        dict_len = len(test_dict)
        batch_index_end = 0

        show_id = np.random.randint(0, dict_len // batch_size-1, 2)

        input_imgs = []
        output_imgs = []
        target_imgs = []

        i = 0
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
                    sample_clean = Image.open(patch_info['path'])
                    if sample_clean.mode == 'RGBA':
                        sample_clean = sample_clean.convert('RGB')

                    sample_processed = AddNoise({'image': sample_clean, 'target': sample_clean})
                    noisy_img = sample_processed['image']
                    clean_img = sample_processed['target']

                    width = patch_info['width']
                    height = patch_info['height']
                    current_im_id = patch_info['im_id']

                p_x1, p_x2, p_y1, p_y2 = patch_info['x1'], patch_info['x2'], patch_info['y1'], patch_info['y2']


                image = noisy_img.crop((p_x1, p_y1, p_x2, p_y2))
                target = clean_img.crop((p_x1, p_y1, p_x2, p_y2))
                if cfg.DATASET.TO_GRAY:
                    image, target = image.convert('L'), target.convert('L')

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
            output = model(images)

            if show_img and i in show_id:
                input_img, output_img, target_img = tensor2img(images, output, targets)
                input_imgs.append(input_img)
                output_imgs.append(output_img.cpu())
                target_imgs.append(target_img)

            for j in range(images.size(0)):
                if not (i == 0 and j == 0) and im_id[j] != previous_im_id:
                    im_result, gt_result = joint_patches(output_buffer, previous_im_w, previous_im_h, cfg.MODEL.IN_CHANNEL)
                    im_result[im_result > 1.0] = 1.0
                    im_result[im_result < 0.0] = 0.0

                    metric_SSIM(im_result.cuda(), gt_result.cuda())
                    metric_PSNR(im_result, gt_result)
                    im_PIL = Image.fromarray(np.array(im_result.squeeze() * 255, np.uint8))
                    im_PIL.save(os.path.join(result_save_dir, previous_im_id))
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

            i += 1

        im_result, gt_result = joint_patches(output_buffer, previous_im_w, previous_im_h, cfg.MODEL.IN_CHANNEL)
        im_result[im_result > 1.0] = 1.0
        im_result[im_result < 0.0] = 0.0
        metric_SSIM(im_result.cuda(), gt_result.cuda())
        metric_PSNR(im_result, gt_result)
        im_PIL = Image.fromarray(np.array(im_result.squeeze() * 255, np.uint8))
        im_PIL.save(os.path.join(result_save_dir, previous_im_id))

    ssim = metric_SSIM.metric_get()
    psnr = metric_PSNR.metric_get()

    if show_img:
        return ssim, psnr, metric_SSIM.im_count, truncated(torch.cat(input_imgs, 2), 1.0, 0.0), \
               truncated(torch.cat(output_imgs, 2), 1.0, 0.0), \
               torch.cat(target_imgs, 2)
    else:
        return ssim, psnr, metric_SSIM.im_count


def denoise_inference_joint(model, test_set_list, cfg, show_img=False, tag='search'):
    logger = logging.getLogger("one_stage_nas.inference")
    print('load test set')

    ssim_list = []
    psnr_list = []
    dataset_size_list = []
    dataset_list = cfg.DATASET.TRAIN_DATASETS
    test_id = random.randint(0, len(dataset_list)-1)

    for id, [dataset, test_list] in enumerate(zip(dataset_list, test_set_list)):
        if dataset in ['CBD_syn']:
            if show_img and id == test_id:
                ssim, psnr, count, input_img, output_img, target_img = \
                    CBD_syn_denoise_inference(model, test_list, cfg, show_img=True, tag=tag)
            else:
                ssim, psnr, count = \
                    CBD_syn_denoise_inference(model, test_list, cfg, show_img=False, tag=tag)

        elif dataset in ['CBD_real']:
            if show_img and id == test_id:
                ssim, psnr, count, input_img, output_img, target_img = \
                    CBD_real_denoise_inference(model, test_list, cfg, show_img=True, tag=tag)
            else:
                ssim, psnr, count = \
                    CBD_real_denoise_inference(model, test_list, cfg, show_img=False, tag=tag)

        ssim_list.append(ssim)
        psnr_list.append(psnr)
        dataset_size_list.append(count)
        # dataset_size_list = [1.0, 0.0]
        im_count = np.array(dataset_size_list).sum()
        # im_count = [1, 0]

    logger_info = []
    ssim_combine = 0
    psnr_combine = 0
    for set_id, [ssim, psnr, count] in enumerate(zip(ssim_list, psnr_list, dataset_size_list)):
        logger_info.append('Val_dataset_{} SSIM:{} PSNR:{}'.format(set_id, ssim, psnr))
        ssim_combine += ssim*count/im_count
        psnr_combine += psnr*count/im_count
    logger_info.append('Val SSIM:{} PSNR:{}'.format(ssim_combine, psnr_combine))

    logger.info('  '.join(logger_info))

    if show_img:
        return ssim_combine, psnr_combine, input_img, output_img, target_img
    else:
        return ssim_combine, psnr_combine


def derain_inference(model, test_list, cfg, show_img=False, tag='search'):
    logger = logging.getLogger("one_stage_nas.inference")
    print('load test set')

    crop_size = cfg.DATASET.CROP_SIZE
    data_root = cfg.DATASET.DATA_ROOT

    test_dict = []
    for im_info in test_list:

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

    print('evaluation')

    transforms = build_transforms(task='derain', tag='test')

    model.eval()
    metric_SSIM = SSIM(window_size=11, channel=cfg.MODEL.IN_CHANNEL, is_cuda=True)
    metric_PSNR = PSNR()

    batch_size = cfg.DATALOADER.BATCH_SIZE_TEST

    if tag == 'search':
        result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                    '{}_{}'.format(cfg.DATASET.DATA_NAME, cfg.RESULT_DIR), 'search/img_result'))
    elif tag == 'train':
        result_save_dir = '/'.join((cfg.OUTPUT_DIR, cfg.DATASET.TASK,
                                    '{}_{}'.format(cfg.DATASET.DATA_NAME, cfg.RESULT_DIR),
                                    'train/img_result'))
    mkdir(result_save_dir)

    with torch.no_grad():
        previous_im_id = ''
        current_im_id = ''
        previous_im_w = None
        previous_im_h = None
        output_buffer = []

        dict_len = len(test_dict)
        batch_index_end = 0

        show_id = np.random.randint(0, dict_len // batch_size - 1, 2)

        input_imgs = []
        output_imgs = []
        target_imgs = []

        i = 0
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
            output = model(images)

            if show_img and i in show_id:
                input_img, output_img, target_img = tensor2img(images, output, targets)
                input_imgs.append(input_img)
                output_imgs.append(output_img.cpu())
                target_imgs.append(target_img)

            for j in range(images.size(0)):
                if not (i == 0 and j == 0) and im_id[j] != previous_im_id:
                    im_result, gt_result = joint_patches(output_buffer, previous_im_w, previous_im_h)
                    im_result[im_result > 1.0] = 1.0
                    im_result[im_result < 0.0] = 0.0

                    metric_SSIM(im_result.cuda(), gt_result.cuda())
                    metric_PSNR(im_result, gt_result)
                    im_PIL = Image.fromarray(np.array(im_result.squeeze() * 255, np.uint8))
                    im_PIL.save(os.path.join(result_save_dir, previous_im_id))
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

            i += 1

        im_result, gt_result = joint_patches(output_buffer, previous_im_w, previous_im_h)
        im_result[im_result > 1.0] = 1.0
        im_result[im_result < 0.0] = 0.0
        metric_SSIM(im_result.cuda(), gt_result.cuda())
        metric_PSNR(im_result, gt_result)
        im_PIL = Image.fromarray(np.array(im_result.squeeze() * 255, np.uint8))
        im_PIL.save(os.path.join(result_save_dir, previous_im_id))

    ssim = metric_SSIM.metric_get()
    psnr = metric_PSNR.metric_get()

    logger.info(' Val: SSIM:{} PSNR:{}'.format(ssim, psnr))

    if show_img:
        return ssim, psnr, truncated(torch.cat(input_imgs, 2), 1.0, 0.0), \
               truncated(torch.cat(output_imgs, 2), 1.0, 0.0), \
               torch.cat(target_imgs, 2)
    else:
        return ssim, psnr
