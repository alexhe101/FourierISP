from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import os
import torch
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop,numpyaug
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import imageio
import numpy as np
import cv2
import colour_demosaicing
import glob
import random
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from concurrent.futures import ThreadPoolExecutor,as_completed
import pickle
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file,tripled_paths_from_lmdb

# 定义一个函数，用于构建lq_path_map的一部分映射

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out
def remove_black_level(img, black_lv=512, white_lv=16383):
    img = np.maximum(img.astype(np.float32)-black_lv, 0) / (white_lv-black_lv)
    return img
def get_raw_demosaic(raw, pattern='BGGR'):  # HxW
    raw_demosaic = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(raw, pattern=pattern)
    raw_demosaic = np.ascontiguousarray(raw_demosaic.astype(np.float32))
    return raw_demosaic  # 3xHxW

@DATASET_REGISTRY.register()
class SSIDRawImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(SSIDRawImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder,self.mask_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt','mask']
            self.paths = tripled_paths_from_lmdb([self.lq_folder, self.gt_folder,self.mask_folder], ['lq', 'gt','mask'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
        if self.opt['phase'] == 'val':
            # random.shuffle(self.paths)  # 随机打乱顺序
            self.paths = self.paths[:1000]  # 仅保留前1000
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt') #rgbdata
        img_gt = imfrombytes(img_bytes, float32=False,flag='unchanged')/65535
        ratio = int(os.path.basename(gt_path)[-7:-4])

        lq_path = self.paths[index]['lq_path']
        img_bytes_lq = self.file_client.get(lq_path, 'lq')
        raw_image = imfrombytes(img_bytes_lq, float32=False,flag='unchanged')

        norm = remove_black_level(raw_image)
        img_combined = pack_raw(norm)*ratio
        if self.opt['phase'] == 'train':
            img_gt,img_combined = numpyaug( [img_gt,img_combined], self.opt['use_hflip'], self.opt['use_rot'])
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        img_combined = img_combined.transpose((2, 0, 1))
        img_combined = torch.from_numpy(img_combined)
        return {'lq': img_combined, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
