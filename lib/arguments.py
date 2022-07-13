import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from lib.utils.common_utils import *

class Arguments:
    def __init__(self, stage='pretrain'):
        self._parser = argparse.ArgumentParser(description='Arguments for pretain|inversion|eval_treegan|eval_completion.')
        
        
        # common args
        self._parser.add_argument('--name', type=str, required=True, help='job name')
        self._parser.add_argument('--data_dir', type=str, default='./datasets/cub', help='dataset dir')
        self._parser.add_argument('--dataset', type=str, default='cub', help='dataset')
        self._parser.add_argument('--gpu_ids', type=str, default='0', help='comma-separated')
        self._parser.add_argument('--tensorboard', action='store_true', default=True)
        self._parser.add_argument('--num_workers', type=int, default=4) 
        
        # GAN Model settings
        self._parser.add_argument('--texture_resolution', type=int, default=512)
        self._parser.add_argument('--mesh_resolution', type=int, default=32)
        self._parser.add_argument('--symmetric_g', type=bool, default=True)
        self._parser.add_argument('--texture_only', action='store_true')
        self._parser.add_argument('--conditional_class', action='store_true', default=True, help='condition the model on class labels')
        self._parser.add_argument('--n_classes', type=int, default=[200], nargs='+', help='CUB has 200 classes')
        self._parser.add_argument('--conditional_color', action='store_true', help='condition the model on colors (p3d only)')
        self._parser.add_argument('--conditional_text', action='store_true', help='condition the model on captions (cub only)')
        self._parser.add_argument('--conditional_encoding', action='store_true', help='condition the model on image encoding')
        self._parser.add_argument('--norm_g', type=str, default='syncbatch', help='(syncbatch|batch|instance|none)')
        self._parser.add_argument('--latent_dim', type=int, default=64, help='dimensionality of the random vector z')
        self._parser.add_argument('--mesh_path', type=str, default='lib/mesh_templates/uvsphere_16rings.obj', help='path to the .obj mesh template')
        self._parser.add_argument('--text_max_length', type=int, default=18)
        self._parser.add_argument('--text_pretrained_encoder', type=str, default='cache/cub/text_encoder200.pth')
        self._parser.add_argument('--text_train_encoder', action='store_true') # Disabled by default (unstable)
        self._parser.add_argument('--text_attention', type=bool, default=True)
        self._parser.add_argument('--text_embedding_dim', type=int, default=256)

        if stage == 'inversion':
            self.parse_inversion_args()

        if stage == 'pretraining':
            self.parse_pretrain_args()
            
        if stage == 'evaluation':
            self.parse_inversion_args()

            self._parser.add_argument('--eval_option', required=True, help='IoU|FID_1|FID_10|FID_12')
            self._parser.add_argument('--canonical_pose', action='store_true', default=True, help='')
            self._parser.add_argument('--default_orientation', type=int, default=30)
            self._parser.add_argument('--default_scale', type=float, default=0.7)
            self._parser.add_argument('--angle_interval', type=int, default=30)
    
    def parse_pretrain_args(self):
        # Training / eval settings
        self._parser.add_argument('--batch_size', type=int, default=32)
        self._parser.add_argument('--continue_train', action='store_true', help='resume training from checkpoint')
        self._parser.add_argument('--epochs', type=int, default=600)
        self._parser.add_argument('--norm_d', type=str, default='none', help='(instance|none)')
        self._parser.add_argument('--mesh_regularization', type=float, default=0.0001, help='strength of the smoothness regularizer')
        self._parser.add_argument('--lr_g', type=float, default=0.0001)
        self._parser.add_argument('--lr_d', type=float, default=0.0004)
        self._parser.add_argument('--d_steps_per_g', type=int, default=2)
        self._parser.add_argument('--g_running_average_alpha', type=float, default=0.999)
        self._parser.add_argument('--lr_decay_after', type=int, default=1000) # Set to a very large value to disable
        self._parser.add_argument('--loss', type=str, default='hinge', help='(hinge|ls|original)')
        self._parser.add_argument('--mask_output', type=bool, default=True)
        self._parser.add_argument('--num_discriminators', type=int, default=-1) # -1 = autodetect
        self._parser.add_argument('--img_gan_loss_wt', type=float, default=0.2, help='weight for 2D discriminator')
        self._parser.add_argument('--patchGAN_apply_mask', action='store_true', help='for patchGAN discriminator')
        self._parser.add_argument('--lock_2d_loss_for_generator', action='store_true', help='for patchGAN discriminator')
        self._parser.add_argument('--checkpoint_freq', type=int, default=20, help='save checkpoint every N epochs')
        self._parser.add_argument('--save_freq', type=int, default=5, help='save latest checkpoint every N epochs')
        self._parser.add_argument('--evaluate_freq', type=int, default=20, help='evaluate FID every N epochs')
        self._parser.add_argument('--evaluate', action='store_true', help='evaluate FID, do not train')
        self._parser.add_argument('--export_sample', action='store_true', help='export image/mesh samples, do not train')
        self._parser.add_argument('--which_epoch', type=str, default='latest', help='(N|latest|best)') 
        self._parser.add_argument('--truncation_sigma', type=float, default=-1, help='-1 = autodetect; set to a large value to disable')
        self._parser.add_argument('--render_res', type=int, default=256, help='render_res')

        # patchD related
        self._parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        self._parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        self._parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        self._parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        self._parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        self._parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        self._parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        self._parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        self._parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        self._parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        self._parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
            
    def parse_inversion_args(self):
        # data related
        self._parser.add_argument('--batch_size', type=int, default=1)
        self._parser.add_argument('--img_size', type=int, default=256)
        self._parser.add_argument('--padding_frac', type=float, default=0.05)
        self._parser.add_argument('--jitter_frac', type=float, default=0.05)
        self._parser.add_argument('--split', type=str, default='test', help=['train', 'val', 'all', 'test'])
        self._parser.add_argument('--shuffle', action='store_true', default=False, help='if test, choose not shuffle')
        self._parser.add_argument('--num_kps', type=int, default=15)
        self._parser.add_argument('--no_mirror', action='store_true', help='flip the image')
        self._parser.add_argument('--target_mask_threshold', type=float, default=0.95, help='the threshold predefine the scaled 0,1 mask of target image')

        # mesh-inversion related
        self._parser.add_argument('--checkpoint_dir', type=str, default='pretrained', help='checkpoint for pretrained GAN')
        self._parser.add_argument('--select_num', type=int, default=500, help='Number of point clouds pool to select from init.')
        self._parser.add_argument('--init_batch_size', type=int, default=1, help='init batch size')
        self._parser.add_argument('--iterations', type=int, default=[50, 50, 50, 50], nargs='+', help='') 
        self._parser.add_argument('--warm_up', type=int, default=0, help='Number of warmup iterations.')
        self._parser.add_argument('--z_lrs', type=float, default=[1e-1, 5e-2, 1e-2, 5e-3], nargs='+', help='learning rate for z')
        self._parser.add_argument('--use_pred_pose', action='store_true', default=True, help='if use poses pred by CMR')
        self._parser.add_argument('--filter_noisy_pred_pose', type=float, default=0.7, help='To filter out highly inaccurate pred poses, with iou < 0.7') 
        self._parser.add_argument('--update_pose', action='store_true', default=True, help='if otptimize poses')
        self._parser.add_argument('--pose_lrs', type=float, default=[1e-2, 5e-3, 1e-3, 5e-4], nargs='+', help='learning rate for pose')
        self._parser.add_argument('--use_predicted_mask', action='store_true', default=True, help='If use predicted mask or GT mask.')
        self._parser.add_argument('--save_results', action='store_true', default=True,  help='save z, 3D model, img, and mask')

        # loss related
        self._parser.add_argument('--chamfer_mask_loss', action='store_true', default=True, help='if use Chamfer mask loss')
        self._parser.add_argument('--chamfer_mask_loss_wt', type=float, default=10.0)
        self._parser.add_argument('--chamfer_texture_pixel_loss', action='store_true', default=True, help='Chamfer texture loss - pixel level')
        self._parser.add_argument('--chamfer_texture_pixel_loss_wt', type=float, default=1.0)
        self._parser.add_argument('--chamfer_texture_feat_loss', action='store_true', default=True, help='Chamfer texture loss - feature level')
        self._parser.add_argument('--chamfer_texture_feat_loss_wt', type=float, default=0.04)
        self._parser.add_argument('--xy_threshold', type=float, default=0.16)
        self._parser.add_argument('--xy_k', type=float, default=1.0)
        self._parser.add_argument('--xy_alpha', type=float, default=1)
        self._parser.add_argument('--rgb_eps', type=float, default=1)
        self._parser.add_argument('--subpool_threshold', type=float, default=0.5)
        self._parser.add_argument('--chamfer_resolution', type=int, default=8192, help='resolution for computing chamfer texture losses')         
        # other losses
        self._parser.add_argument('--mesh_regularization_loss', action='store_true', default=False, help='')
        self._parser.add_argument('--mesh_regularization_loss_wt', type=float, default=0.00005)
        self._parser.add_argument('--nll_loss', action='store_true', default=True, help='')
        self._parser.add_argument('--nll_loss_wt', type=float, default=0.05)

    
    def parser(self):
        return self._parser
    


    
   


