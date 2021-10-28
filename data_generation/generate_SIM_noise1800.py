from glob import glob
import numpy as np
import subprocess
import os

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

source_dir = '/home/hkzhang/Documents/sdb_a/real_noise/clean_training_patches/images'
save_dir = '/home/hkzhang/Documents/sdb_a/nas_data/denoise/SIM_noise1800'

noise_image_list = glob(source_dir + '/noise_img/*_noise.jpg')
clean_image_list = glob(source_dir + '/clean_img/*.jpg')

noise_image_list.sort()
clean_image_list.sort()

# noise_image_id_list = [item.split('/')[-1][:-10] for item in noise_image_list]
#
# for item in clean_image_list:
#     clean_id = item.split('/')[-1][:-4]
#     if not clean_id in noise_image_id_list:
#         os.remove(item)


train_clean_save_dir = save_dir + '/train/clean_img/'
train_noise_save_dir = save_dir + '/train/noise_img/'

test_clean_save_dir = save_dir + '/test/clean_img/'
test_noise_save_dir = save_dir + '/test/noise_img/'

make_if_not_exist(train_clean_save_dir)
make_if_not_exist(train_noise_save_dir)
make_if_not_exist(test_clean_save_dir)
make_if_not_exist(test_noise_save_dir)

for i, [clean_dir, noise_dir] in enumerate(zip(clean_image_list[:1600], noise_image_list[:1600])):

    clean_save_dir = train_clean_save_dir + '%04d' % i + '.jpg'
    noise_save_dir = train_noise_save_dir + '%04d' % i + '.jpg'

    clean_save_command = 'cp {} {}'.format(clean_dir, clean_save_dir)
    noise_save_command = 'cp {} {}'.format(noise_dir, noise_save_dir)

    subprocess.call(clean_save_command, shell=True)
    subprocess.call(noise_save_command, shell=True)

for i, [clean_dir, noise_dir] in enumerate(zip(clean_image_list[1600:], noise_image_list[1600:])):

    clean_save_dir = test_clean_save_dir + '%04d' % i + '.jpg'
    noise_save_dir = test_noise_save_dir + '%04d' % i + '.jpg'

    clean_save_command = 'cp {} {}'.format(clean_dir, clean_save_dir)
    noise_save_command = 'cp {} {}'.format(noise_dir, noise_save_dir)

    subprocess.call(clean_save_command, shell=True)
    subprocess.call(noise_save_command, shell=True)




