import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os
import sys
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from option import opt

# ---------------- BASIC CONFIG ----------------
BS = opt.bs
crop_size = 'whole_img'
if opt.crop:
    crop_size = opt.crop_size


# ---------------- RESIDE DATASET ----------------
class RESIDE_Dataset(data.Dataset):
    def __init__(self, path, train=True, size=crop_size, format='.png'):
        super().__init__()
        self.size = size
        self.train = train
        self.format = format

        self.haze_dir = os.path.join(path, 'hazy')
        self.clear_dir = os.path.join(path, 'clear')

        self.haze_imgs = sorted(os.listdir(self.haze_dir))
        self.haze_imgs = [os.path.join(self.haze_dir, img) for img in self.haze_imgs]

        print(f"[RESIDE] Loaded {len(self.haze_imgs)} images from {path}")
        print("crop size:", size)

    def __len__(self):
        return len(self.haze_imgs)

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index]).convert("RGB")

        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, len(self.haze_imgs) - 1)
                haze = Image.open(self.haze_imgs[index]).convert("RGB")

        img_name = os.path.basename(self.haze_imgs[index])
        img_id = img_name.split('_')[0]
        clear_name = img_id + self.format
        clear = Image.open(os.path.join(self.clear_dir, clear_name)).convert("RGB")

        clear = tfs.CenterCrop(haze.size[::-1])(clear)

        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, (self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)

        haze, clear = self.augment(haze, clear)
        return haze, clear

    def augment(self, haze, clear):
        if self.train:
            if random.random() > 0.5:
                haze = FF.hflip(haze)
                clear = FF.hflip(clear)
            rot = random.randint(0, 3)
            if rot:
                haze = FF.rotate(haze, 90 * rot)
                clear = FF.rotate(clear, 90 * rot)

        haze = tfs.ToTensor()(haze)
        haze = tfs.Normalize(
            mean=[0.64, 0.6, 0.58],
            std=[0.14, 0.15, 0.152]
        )(haze)

        clear = tfs.ToTensor()(clear)
        return haze, clear


# ---------------- SMOKEBENCH DATASET ----------------
class SmokeBenchDataset(data.Dataset):
    def __init__(self, root, mode='Train'):
        self.hazy_dir = os.path.join(root, mode, 'hazy')
        self.clear_dir = os.path.join(root, mode, 'clear')

        self.files = sorted(os.listdir(self.hazy_dir))

        self.transform_haze = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.64, 0.6, 0.58],
                          std=[0.14, 0.15, 0.152])
        ])

        self.transform_clear = tfs.ToTensor()

        print(f"[SmokeBench] {mode} set: {len(self.files)} images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        hazy = Image.open(os.path.join(self.hazy_dir, fname)).convert('RGB')
        clear = Image.open(os.path.join(self.clear_dir, fname)).convert('RGB')

        return self.transform_haze(hazy), self.transform_clear(clear)


# ---------------- LAZY LOADERS (CRITICAL FIX) ----------------
def ITS_train_loader(opt):
    return DataLoader(
        RESIDE_Dataset(
            os.path.join(opt.data_dir, 'RESIDE/ITS'),
            train=True,
            size=crop_size
        ),
        batch_size=opt.bs,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True
    )


def ITS_test_loader(opt):
    return DataLoader(
        RESIDE_Dataset(
            os.path.join(opt.data_dir, 'RESIDE/SOTS/indoor'),
            train=False,
            size='whole_img'
        ),
        batch_size=1,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=True
    )


def OTS_train_loader(opt):
    return DataLoader(
        RESIDE_Dataset(
            os.path.join(opt.data_dir, 'RESIDE/OTS'),
            train=True,
            format='.jpg'
        ),
        batch_size=opt.bs,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True
    )


def OTS_test_loader(opt):
    return DataLoader(
        RESIDE_Dataset(
            os.path.join(opt.data_dir, 'RESIDE/SOTS/outdoor'),
            train=False,
            size='whole_img',
            format='.png'
        ),
        batch_size=1,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=True
    )


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
