import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0.,), std=(1.,)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        if len(sample) > 1:
            mask = sample['label']
            mask = np.array(mask).astype(np.float32)
            return {'image': img,
                    'label': mask}

        return {'image': img}


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        img = img.resize((self.size, self.size))

        if len(sample) > 1:
            mask = sample['label']
            mask = mask.resize((self.size, self.size))
            return {'image': img,
                    'label': mask}

        return {'image': img}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        if len(sample) > 1:
            mask = sample['label']
            mask = np.array(mask).astype(np.float32)
            mask = torch.from_numpy(mask).float()
            return {'image': img,
                    'label': mask}

        return {'image': img}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']

        if len(sample) > 1:
            mask = sample['label']
            if random.random() < 0.7:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            return {'image': img,
                    'label': mask}

        else:
            if random.random() < 0.7:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return {'image': img}


class RandomVerticalFlip(object):
    def __call__(self, sample):
        img = sample['image']

        if len(sample) > 1:
            mask = sample['label']
            if random.random() < 0.7:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            return {'image': img,
                    'label': mask}

        else:
            if random.random() < 0.7:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return {'image': img}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = sample['image']
        img = img.rotate(rotate_degree, Image.BILINEAR)

        if len(sample) > 1:
            mask = sample['label']
            mask = mask.rotate(rotate_degree, Image.NEAREST)
            return {'image': img,
                    'label': mask}

        return {'image': img}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        if len(sample) > 1:
            mask = sample['label']
            return {'image': img,
                    'label': mask}

        return {'image': img}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = None
        if len(sample) > 1:
            mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.6), int(self.base_size * 1.2))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        if mask:
            mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            if mask:
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        nw, nh = img.size
        x1 = random.randint(0, nw - self.crop_size)
        y1 = random.randint(0, nh - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        if mask:
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

            return {'image': img,
                    'label': mask}

        return {'image': img}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = None
        if len(sample) > 1:
            mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        if mask:
            mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        if mask:
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

            return {'image': img,
                    'label': mask}

        return {'image': img}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        img = img.resize(self.size, Image.BILINEAR)

        if len(sample) > 1:
            mask = sample['label']
            mask = mask.resize(self.size, Image.NEAREST)
            return {'image': img,
                    'label': mask}

        return {'image': img}


class AddAxis(object):
    def __call__(self, sample):
        img = sample['image']
        img = np.array(img, dtype=np.float32)
        img = img[:, :, np.newaxis]

        if len(sample) > 1:
            mask = sample['label']
            return {'image': img,
                    'label': mask}

        return {'image': img}


class ToRGB(object):
    def __call__(self, sample):
        im = sample['image']
        img = Image.merge('RGB', (im, im, im))

        if len(sample) > 1:
            mask = sample['label']
            return {'image': img,
                    'label': mask}

        return {'image': img}
