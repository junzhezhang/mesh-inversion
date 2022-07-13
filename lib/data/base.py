"""
Base data loading class.

Should output:
    - img: B X 3 X H X W
    - kp: B X nKp X 2
    - mask: B X H X W
    - sfm_pose: B X 7 (s, tr, q)
    (kp, sfm_pose) correspond to image coordinates in [-1, 1]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np

import scipy.linalg
import scipy.ndimage.interpolation
from skimage.io import imread
from absl import flags, app

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from lib.utils import image as image_utils
from lib.utils import transformations
from lib.utils.inversion_dist import *
import cv2
import os

# -------------- Dataset ------------- #
# ------------------------------------ #
class BaseDataset(Dataset):
    ''' 
    img, mask, kp, pose data loader
    '''

    def __init__(self, args, filter_key=None):
        self.args = args
        self.img_size = args.img_size
        self.jitter_frac = args.jitter_frac
        self.padding_frac = args.padding_frac
        self.filter_key = filter_key

        
            

    def forward_img(self, index):
        data = self.anno[index]
        data_sfm = self.anno_sfm[index]

        if self.args.use_predicted_mask:
            pred_mask_path = os.path.join(self.pred_mask_dir,str(index)+'.npy')
            pred_mask = np.load(pred_mask_path,allow_pickle=True)

            ### IoU
            input_mask = data.mask
            overlap_mask = input_mask * pred_mask
            union_mask = input_mask + pred_mask - overlap_mask
            iou = overlap_mask.sum()/union_mask.sum()
            
            ### replace
            data.mask = pred_mask

        # sfm_pose = (sfm_c, sfm_t, sfm_r)
        sfm_pose = [np.copy(data_sfm.scale), np.copy(data_sfm.trans), np.copy(data_sfm.rot)]

        sfm_rot = np.pad(sfm_pose[2], (0,1), 'constant')
        sfm_rot[3, 3] = 1
        sfm_pose[2] = transformations.quaternion_from_matrix(sfm_rot, isprecise=True)

        if self.args.dataset == 'p3d':
            # NOTE: temp fix car_imagenet\\n02814533_4600.JPEG
            sub_dirs = data.rel_path.split('\\')
            data.rel_path = sub_dirs[0]+'/'+sub_dirs[1]
        img_path = osp.join(self.img_dir, str(data.rel_path))

        img = imread(img_path) / 255.0
        # Some are grayscale:
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        mask = np.expand_dims(data.mask, 2)


        # Adjust to 0 indexing
        bbox = np.array(
            [data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2],
            float) - 1

        parts = data.parts.T.astype(float)
        kp = np.copy(parts)
        vis = kp[:, 2] > 0
        kp[vis, :2] -= 1

        # Peturb bbox
        if self.args.split == 'train':
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=self.jitter_frac)
        else:
            bbox = image_utils.peturb_bbox(
                bbox, pf=self.padding_frac, jf=0)
        bbox = image_utils.square_bbox(bbox)

        # crop image around bbox, translate kps
        img, mask, kp, sfm_pose = self.crop_image(img, mask, bbox, kp, vis, sfm_pose)
        # set(data.mask.reshape(-1).tolist()) --> {0,1} I # NOTE: mask error is in below
        # scale image, and mask. And scale kps.        
        img, mask, kp, sfm_pose = self.scale_image(img, mask, kp, vis, sfm_pose)

        # Mirror image on random.
        if self.args.split == 'train' and (not self.args.no_mirror):
            img, mask, kp, sfm_pose = self.mirror_image(img, mask, kp, sfm_pose)

        # Normalize kp to be [-1, 1]
        img_h, img_w = img.shape[:2]
        kp_norm, sfm_pose = self.normalize_kp(kp, sfm_pose, img_h, img_w)

        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))

        return img, kp_norm, mask, sfm_pose, img_path

    def normalize_kp(self, kp, sfm_pose, img_h, img_w):
        vis = kp[:, 2, None] > 0
        new_kp = np.stack([2 * (kp[:, 0] / img_w) - 1,
                           2 * (kp[:, 1] / img_h) - 1,
                           kp[:, 2]]).T
        sfm_pose[0] *= (1.0/img_w + 1.0/img_h)
        sfm_pose[1][0] = 2.0 * (sfm_pose[1][0] / img_w) - 1
        sfm_pose[1][1] = 2.0 * (sfm_pose[1][1] / img_h) - 1
        new_kp = vis * new_kp

        return new_kp, sfm_pose

    def crop_image(self, img, mask, bbox, kp, vis, sfm_pose):
        # crop image and mask and translate kps
        img = image_utils.crop(img, bbox, bgval=1)
        mask = image_utils.crop(mask, bbox, bgval=0)
        kp[vis, 0] -= bbox[0]
        kp[vis, 1] -= bbox[1]
        sfm_pose[1][0] -= bbox[0]
        sfm_pose[1][1] -= bbox[1]
        return img, mask, kp, sfm_pose

    def scale_image(self, img, mask, kp, vis, sfm_pose):
        # Scale image so largest bbox size is img_size
        bwidth = np.shape(img)[0]
        bheight = np.shape(img)[1]
        scale = self.img_size / float(max(bwidth, bheight))
        
        img_scale, _ = image_utils.resize_img(img, scale)
        mask_scale, _ = image_utils.resize_img(mask, scale) # NOTE bug is here

        kp[vis, :2] *= scale
        sfm_pose[0] *= scale
        sfm_pose[1] *= scale
             
        mask_scale04 = ((mask_scale > self.args.target_mask_threshold) * 1).astype('float64')

        return img_scale, mask_scale04, kp, sfm_pose

    def mirror_image(self, img, mask, kp, sfm_pose):
        kp_perm = self.kp_perm
        if np.random.rand(1) > 0.5:
            # Need copy bc torch collate doesnt like neg strides
            img_flip = img[:, ::-1, :].copy()
            mask_flip = mask[:, ::-1].copy()
            
            # Flip kps.
            new_x = img.shape[1] - kp[:, 0] - 1
            kp_flip = np.hstack((new_x[:, None], kp[:, 1:]))
            kp_flip = kp_flip[kp_perm, :]
            # Flip sfm_pose Rot.
            R = transformations.quaternion_matrix(sfm_pose[2])
            flip_R = np.diag([-1, 1, 1, 1]).dot(R.dot(np.diag([-1, 1, 1, 1])))
            sfm_pose[2] = transformations.quaternion_from_matrix(flip_R, isprecise=True)
            # Flip tx
            tx = img.shape[1] - sfm_pose[1][0] - 1
            sfm_pose[1][0] = tx
            return img_flip, mask_flip, kp_flip, sfm_pose
        else:
            return img, mask, kp, sfm_pose

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        self.index = index
        img, kp, mask, sfm_pose, img_path = self.forward_img(index)
        sfm_pose[0].shape = 1

        mask_tensor = torch.from_numpy(mask)
        mask_dts = image_utils.compute_dt_barrier(mask_tensor)
        
        basename = os.path.basename(img_path)

        category = np.array([self.basename_to_class[basename]])
        
        elem = {
            'idx': index,
            'img': img,
            'kp': kp,
            'mask': mask,
            'sfm_pose': np.concatenate(sfm_pose),
            'inds': index,
            'img_path': img_path,
            'mask_dt': mask_dts,
            'class': category,
        }

        if self.filter_key is not None:
            if self.filter_key not in elem.keys():
                print('Bad filter key %s' % self.filter_key)
            if self.filter_key == 'sfm_pose':
                # Return both vis and sfm_pose
                vis = elem['kp'][:, 2]
                elem = {
                    'vis': vis,
                    'sfm_pose': elem['sfm_pose'],
                }
            else:
                elem = elem[self.filter_key]

        return elem

# ------------ Data Loader ----------- #
# ------------------------------------ #
def base_loader(d_set_func, batch_size, args, filter_key=None, shuffle=True):
    if args.dataset == 'cub':
        dset = d_set_func(args, filter_key=filter_key)
    elif args.dataset == 'p3d':
        dset = d_set_func(args)
    else:
        raise
    try:
        sampler = DistributedSampler(dset) if args.dist else None
        if args.dist:
            shuffle = False
    except:
        sampler = None
        
    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=False)
