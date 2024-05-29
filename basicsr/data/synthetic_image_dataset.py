import cv2
import time
from torch.utils import data as data

from basicsr.data.data_util import paired_paths_from_folder
from basicsr.data.ISP_implement import ISP
from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, scandir


class Dataset_SyntheticDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read GT image and synthesis real noisy srgb image.
    """

    def __init__(self, opt):
        super(Dataset_SyntheticDenoising, self).__init__()
        self.opt = opt

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'   

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)

        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(gt_path))

        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

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
        return len(self.paths)