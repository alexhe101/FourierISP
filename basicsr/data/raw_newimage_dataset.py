from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import random
import torch
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop,numpyaug
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import imageio
import numpy as np
import cv2
import colour_demosaicing
m_blk = 255
m_wlk = 4095
huawei_blk = 63
huawei_wlk = 4*255
def remove_black_level(img, black_lv=huawei_blk, white_lv=huawei_wlk):
    img = np.maximum(img.astype(np.float32)-black_lv, 0) / (white_lv-black_lv)
    return img
def get_raw_demosaic(raw, pattern='RGGB'):  # HxW
    raw_demosaic = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(raw, pattern=pattern)
    raw_demosaic = np.ascontiguousarray(raw_demosaic.astype(np.float32))
    return raw_demosaic  # 3xHxW
def extract_bayer_channels(raw):
    # Reshape the input bayer image
    ch_B  = raw[1::2, 1::2] #1001 RGGB 0011 GGRB
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = np.maximum(RAW_combined.astype(np.float32)-63, 0) / (4 * 255-63)
    # RAW_norm = np.maximum(RAW_combined.astype(np.float32), 0) / (4 * 255)

    return RAW_norm

@DATASET_REGISTRY.register()
class RAWNewImageDataset(data.Dataset):
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
        super(RAWNewImageDataset, self).__init__()
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
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
        # if self.opt['phase'] == 'val':
        #     random.shuffle(self.paths)  # 随机打乱顺序
        #     self.paths = self.paths[:50]  # 仅保留前1000

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        # if self.opt['phase'] != 'train':
        #     random_index = random.randint(0, len(self.paths) - 1)
        #     gt_path = self.paths[random_index]['gt_path']

        img_bytes = self.file_client.get(gt_path, 'gt') #rgbdata
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes_lq = self.file_client.get(lq_path, 'lq')
        raw_image = imfrombytes(img_bytes_lq, float32=False,flag='unchanged')
        raw_image = remove_black_level(imfrombytes(img_bytes_lq, float32=False,flag='unchanged'))
        img_lq = get_raw_demosaic(raw_image)

        # print("image_gt.size:"+str(img_gt.shape))
        # assert(img_lq.shape[2]==4)
        # img_lq = img_lq.transpose((2, 0, 1))
        # img_bytes = self.file_client.get(lq_path, 'lq')
        # img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            # gt_size = self.opt['gt_size']
            # random crop
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt,img_lq = numpyaug( [img_gt,img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_gt.shape[0] * scale, 0:img_gt.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        img_lq = img_lq.transpose((2, 0, 1))
        img_lq = torch.from_numpy(img_lq)
        # img_gt = img_lq
        # for full-resolution
        # img_lq = img_lq.unsqueeze(0)
        # img_lq = F.interpolate(img_lq,scale_factor=0.25,mode='bilinear')
        # img_lq = img_lq.squeeze(0)
        # img_gt = img_lq
        #
        # C,H,W = img_lq.size()
        # nh,nw = H//4,W//4
        # nh,nw = nh//8,nw//8
        # nh,nw = nh*8,nw*8
        # print((nh,nw))
        #         # for full-resolution
        # img_lq = img_lq.unsqueeze(0)
        # img_lq = F.interpolate(img_lq,size=(nh,nw),mode='bilinear')
        # img_lq = img_lq.squeeze(0)
        # img_gt = img_lq

        # img_lq = F.interpolate(img_lq,scale_factor=0.125,mode='bilinear')
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        # print("#############IN RAW PACKED!")
        # print(img_lq.size())
        # print(img_gt.size())
        # return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
