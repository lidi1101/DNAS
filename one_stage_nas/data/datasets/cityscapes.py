import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class CityscapesDataset(Dataset):
    """The classes to id map is copied from
       https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/cityscapes_loader.py
    """

    def __init__(
            self,
            data_file, data_dir,
            transform,
            fg_classes=None, has_flow=False):

        with open(data_file, 'rb') as f:
            datalist = f.readlines()
            self.datalist = [x.decode('utf-8').strip('\n').split('\t') for x in datalist]
            del datalist
        self.root_dir = data_dir
        self.transform = transform
        self.has_flow = has_flow
        self.n_frames = len(self.datalist[0]) // 2  # n images + (n - 1) flos and 1 mask

        # cityscapes
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24,
                              25, 26, 27, 28, 31, 32, 33,]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(19)))

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def set_config(self, crop_size, shorter_side, fg_classes=None):
        try:
            self.transform.transforms[0].shorter_side = shorter_side
            self.transform.transforms[2].crop_size = crop_size
        except AttributeError:
            pass

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        def read_image(x):
            img_arr = np.array(Image.open(x), dtype=np.float32)
            if len(img_arr.shape) == 2: # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            return img_arr
        sample = dict()
        img_name = os.path.join(self.root_dir, self.datalist[idx][0])
        msk_name = os.path.join(self.root_dir, self.datalist[idx][1])
        image = read_image(img_name)
        mask = np.array(Image.open(msk_name))
        mask = self.encode_segmap(mask)
        if img_name != msk_name:
            assert len(mask.shape) == 2, 'Masks must be encoded without colormap'
        sample = {'image': image, 'mask': mask}
        sample = self.transform(sample)
        return sample['image'], sample['mask']
