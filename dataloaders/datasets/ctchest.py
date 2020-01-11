from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr


class CTSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 5
    IN_CHANNELS = 1

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('ctchest'),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(self._base_dir, splt + '.txt'), "r")as f:   # index file.txt
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image, _cat = line.strip().split(' ')
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_train(sample)
            elif split == 'val':
                return self.transform_val(sample)
            elif split == 'test':
                return self.transform_test(sample)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index])
        _target = Image.open(self.categories[index])

        return _img, _target

    def transform_train(self, sample):
        composed_transforms = transforms.Compose([
            #tr.RandomHorizontalFlip(),
            #tr.RandomRotate(10),
            #tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            #tr.RandomGaussianBlur(),
            tr.AddAxis(),
            tr.Normalize(),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            #tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.AddAxis(),
            tr.Normalize(),
            tr.ToTensor()])

        return composed_transforms(sample)
        
    def transform_test(self, sample):
        composed_transforms = transforms.Compose([
            tr.AddAxis(),
            tr.Normalize(),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'CT(split=' + str(self.split) + ')'
