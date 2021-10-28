from PIL import Image
from glob import glob
import numpy as np
import os


source_dir = '/home/hkzhang/Documents/sdb_a/real_rain/compose'
save_dir = '/home/hkzhang/Documents/sdb_a/nas_data/derain/MPID_RS'

clean_list = glob(source_dir + '/ground_truth/*.png')
rainy_im_list = glob(source_dir + '/blended/*.png')

clean_list.sort()
rainy_im_list.sort()

indices = np.random.permutation(len(clean_list))

train_clean_list = [clean_list[indices[i]] for i in range(0, 2400)]
train_rainy_list = [rainy_im_list[indices[i]] for i in range(0, 2400)]

test_clean_list = [clean_list[indices[i]] for i in range(2400, 2600)]
test_rainy_list = [rainy_im_list[indices[i]] for i in range(2400, 2600)]


train_save_dir = save_dir + '/train'
if not os.path.exists(train_save_dir): os.makedirs(train_save_dir)
for i, [clean_im_dir, rainy_im_dir] in enumerate(zip(train_clean_list, train_rainy_list)):
    clean_id = clean_im_dir.split('/')[-1][:-4]
    rainy_id = rainy_im_dir.split('/')[-1][:-4]

    assert clean_id == rainy_id
    print('im: {}'.format(clean_im_dir.split('/')[-1][:-4]))

    im_clean = np.array(Image.open(clean_im_dir))
    im_rain = np.array(Image.open(rainy_im_dir))
    im_save = Image.fromarray(np.hstack(np.stack((im_clean, im_rain))))
    im_save.save(os.path.join(train_save_dir, '{}.jpg'.format(i)))


test_save_dir = save_dir + '/test'
if not os.path.exists(test_save_dir): os.makedirs(test_save_dir)
for i, [clean_im_dir, rainy_im_dir] in enumerate(zip(test_clean_list, test_rainy_list)):
    clean_id = clean_im_dir.split('/')[-1][:-4]
    rainy_id = rainy_im_dir.split('/')[-1][:-4]

    assert clean_id == rainy_id
    print('im: {}'.format(clean_im_dir.split('/')[-1][:-4]))

    im_clean = np.array(Image.open(clean_im_dir))
    im_rain = np.array(Image.open(rainy_im_dir))
    im_save = Image.fromarray(np.hstack(np.stack((im_clean, im_rain))))
    im_save.save(os.path.join(test_save_dir, '{}.jpg'.format(i)))
