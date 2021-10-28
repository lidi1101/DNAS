from PIL import Image
from glob import glob
import numpy as np
import os


source_dir = '/home/hkzhang/Documents/sdb_a/dataset_rain_remove/rain100_h'
save_dir = '/home/hkzhang/Documents/sdb_a/nas_data/derain/rain100_h'

# train_clean_list = glob(source_dir + '/train/norain/*.png')
# train_clean_list.sort()
#
# train_save_dir = save_dir + '/train'
# if not os.path.exists(train_save_dir): os.makedirs(train_save_dir)
# for i, clean_im_dir in enumerate(train_clean_list):
#     print('im: {}'.format(clean_im_dir.split('/')[-1][:-4]))
#     rain_im_dir = os.path.join(source_dir, 'train', 'rain', clean_im_dir.split('/')[-1][:-4]+'x2.png')
#     im_clean = np.array(Image.open(clean_im_dir))
#     im_rain = np.array(Image.open(rain_im_dir))
#     im_save = Image.fromarray(np.hstack(np.stack((im_clean, im_rain))))
#     im_save.save(os.path.join(train_save_dir, '{}.jpg'.format(i)))
#
#
test_clean_list = glob(source_dir + '/test/norain/*.png')
test_clean_list.sort()

test_save_dir = save_dir + '/test'
if not os.path.exists(test_save_dir): os.makedirs(test_save_dir)
for i, clean_im_dir in enumerate(test_clean_list):
    print('im: {}'.format(clean_im_dir.split('/')[-1][:-4]))
    rain_im_dir = os.path.join(source_dir, 'test', 'rain/X2', clean_im_dir.split('/')[-1][:-4]+'x2.png')
    im_clean = np.array(Image.open(clean_im_dir))
    im_rain = np.array(Image.open(rain_im_dir))
    im_save = Image.fromarray(np.hstack(np.stack((im_clean, im_rain))))
    im_save.save(os.path.join(test_save_dir, '{}.jpg'.format(i)))
