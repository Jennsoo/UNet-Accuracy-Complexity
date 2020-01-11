import numpy as np

from dataloaders.datasets import ctchest
from dataloaders.datasets import segthor
from torch.utils.data import DataLoader


def _init_fn():
    np.random.seed(1)

def make_data_loader(args, **kwargs):
    if args.dataset == 'ctchest':
        train_set = ctchest.CTSegmentation(args, split='train')
        val_set = ctchest.CTSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        in_channels = train_set.IN_CHANNELS
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, val_loader, num_class, in_channels

    else:
        raise NotImplementedError

