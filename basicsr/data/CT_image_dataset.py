import numpy as np
import os
import random
import time
import torch
from glob import glob
from torch.utils import data as data
import h5py as h5

from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.utils.img_util import img2tensor


class Dataset_CTDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read GT image and synthesis real noisy srgb image. Online fashion
    """

    def __init__(self, opt):
        super(Dataset_CTDenoising, self).__init__()
        self.opt = opt

        self.gt_folder = opt['dataroot_gt']
        test_patient = opt['test_patient']

        self.target_path = sorted(glob(os.path.join(self.gt_folder, '*target*')))
        self.input_path = sorted(glob(os.path.join(self.gt_folder, '*input*')))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']
            self.sigma_type  = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']

            # input_ = [f for f in self.input_path if test_patient not in f]
            self.target_ = [f for f in self.target_path if test_patient not in f]
        else:
            self.input_ = [f for f in self.input_path if test_patient in f]
            self.target_ = [f for f in self.target_path if test_patient in f]

    def __getitem__(self, index):

        index = index % len(self.target_)

        gt_path = self.target_[index]
        img_gt = np.load(gt_path); img_gt = img_gt[..., np.newaxis]

        if self.opt['phase'] != 'train':
            lq_path = self.input_[index]
            img_lq = np.load(lq_path); img_lq = img_lq[..., np.newaxis]

        if self.opt['phase'] == 'train':
            lq_path = gt_path; img_lq = img_gt.copy()
            gt_size = self.opt['gt_size']
            # padding
            # img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, 1,
                                                'none')
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)

            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value])/255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:
            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)

        return {
                'lq': img_lq,
                'gt': img_gt,
                'lq_path': lq_path,
                'gt_path': gt_path
            }

    def __len__(self):
        return len(self.target_)