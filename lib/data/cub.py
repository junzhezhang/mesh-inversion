"""
CUB has 11788 images total, for 200 subcategories.
5994 train, 5794 test images.

After removing images that are truncated:
min kp threshold 6: 5964 train, 5771 test.
min_kp threshold 7: 5937 train, 5747 test.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np

import scipy.io as sio
from absl import flags, app

import torch
from torch.utils.data import Dataset

from lib.data import base as base_data

# -------------- Dataset ------------- #
# ------------------------------------ #
class CUBDataset(base_data.BaseDataset):
    '''
    CUB Data loader
    '''

    def __init__(self, args, filter_key=None):
        super(CUBDataset, self).__init__(args, filter_key=filter_key)
        self.data_dir = args.data_dir
  
        self.img_dir = osp.join(self.data_dir, 'CUB_200_2011', 'images')
        self.anno_path = osp.join(self.data_dir, 'cache', 'data', '%s_cub_cleaned.mat' % args.split)
        self.anno_sfm_path = osp.join(self.data_dir, 'cache', 'sfm', 'anno_%s.mat' % args.split)

        if self.args.use_predicted_mask:
            self.pred_mask_dir = osp.join(self.data_dir, 'predicted_mask')  

        self.filter_key = filter_key
        
        # Load the annotation file.
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)
        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1;
        
        ### get basename to class dictionary
        # copied from CubPseudoDataset
        # Load CUB labels
        cub_path = 'datasets/cub/CUB_200_2011'
        import os
        with open(os.path.join(cub_path, 'images.txt'), 'r') as f:
            images = f.readlines()
            images = [x.split(' ') for x in images]
            ids = {k: v.strip() for k, v in images}

        with open(os.path.join(cub_path, 'image_class_labels.txt'), 'r') as f:
            classes = f.readlines()
            classes = [x.split(' ') for x in classes]
            classes = {k: int(v.strip())-1 for k, v in classes}

        self.basename_to_class = {}
        for k, c in classes.items():
            fname = ids[k]
            basename = os.path.basename(fname)
            self.basename_to_class[basename] = c

#----------- Data Loader ----------#
#----------------------------------#
def data_loader(args, shuffle=False):
    return base_data.base_loader(CUBDataset, args.batch_size, args, filter_key=None, shuffle=shuffle)


def kp_data_loader(batch_size, args):
    return base_data.base_loader(CUBDataset, batch_size, args, filter_key='kp')


def mask_data_loader(batch_size, args):
    return base_data.base_loader(CUBDataset, batch_size, args, filter_key='mask')

    
def sfm_data_loader(batch_size, args):
    return base_data.base_loader(CUBDataset, batch_size, args, filter_key='sfm_pose')
