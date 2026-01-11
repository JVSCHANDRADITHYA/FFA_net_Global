import os
import random
from PIL import Image

import torch.utils.data as data
from torch.utils.data import DataLoader

import torchvision.transforms as tfs
from torchvision.transforms import functional as FF

from option import opt


class SmokeBenchDataset(data.Dataset):
    def __init__(self, root, mode='Train'):
        self.hazy_dir = os.path.join(root, mode, 'hazy')
        self.clear_dir = os.path.join(root, mode, 'clear')

        self.files = sorted(os.listdir(self.hazy_dir))
        self.train = (mode == 'Train')

        self.crop = opt.crop
        self.crop_size = opt.crop_size

        print(f"[SmokeBench] {mode} set: {len(self.files)} images")
        if self.crop:
            print(f"[SmokeBench] Using random crop: {self.crop_size}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        hazy = Image.open(os.path.join(self.hazy_dir, fname)).convert('RGB')
        clear = Image.open(os.path.join(self.clear_dir, fname)).convert('RGB')

        # --------- RANDOM CROP (CRITICAL) ---------
        if self.crop:
            w, h = hazy.size
            if w < self.crop_size or h < self.crop_size:
                hazy = hazy.resize(
                    (max(w, self.crop_size), max(h, self.crop_size)),
                    Image.BICUBIC
                )
                clear = clear.resize(
                    (max(w, self.crop_size), max(h, self.crop_size)),
                    Image.BICUBIC
                )

            i, j, th, tw = tfs.RandomCrop.get_params(
                hazy, (self.crop_size, self.crop_size)
            )
            hazy = FF.crop(hazy, i, j, th, tw)
            clear = FF.crop(clear, i, j, th, tw)

        # --------- AUGMENTATION ---------
        if self.train:
            if random.random() > 0.5:
                hazy = FF.hflip(hazy)
                clear = FF.hflip(clear)

            rot = random.randint(0, 3)
            if rot:
                hazy = FF.rotate(hazy, 90 * rot)
                clear = FF.rotate(clear, 90 * rot)

        # --------- TO TENSOR ---------
        hazy = tfs.ToTensor()(hazy)
        hazy = tfs.Normalize(
            mean=[0.64, 0.6, 0.58],
            std=[0.14, 0.15, 0.152]
        )(hazy)

        clear = tfs.ToTensor()(clear)

        return hazy, clear


def SmokeBench_train_loader(opt):
    return DataLoader(
        SmokeBenchDataset(opt.data_dir, mode='Train'),
        batch_size=opt.bs,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True
    )


def SmokeBench_test_loader(opt):
    return DataLoader(
        SmokeBenchDataset(opt.data_dir, mode='Test'),
        batch_size=1,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=True
    )
