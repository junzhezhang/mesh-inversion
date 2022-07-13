import os

import torch
import torch.optim
from torch.utils.data import Dataset
from lib.arguments import Arguments

from lib.utils.common_utils import *
from lib.utils.inversion_dist import *

from lib.mesh_inversion import MeshInversion
from scipy.spatial.transform import Rotation
from lib.utils.cam_pose import quaternion_apply
from lib.utils.fid import calculate_stats, calculate_frechet_distance, init_inception, forward_inception_batch
from PIL import Image

import numpy as np
import imageio
import glob
from tqdm import tqdm

def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]

class EvalDataset(Dataset):
    def __init__(self, path, eval_option, skip_angles=False):

        self.eval_option = eval_option
        
        if eval_option == 'FID_1':
            self.files = glob.glob(path+'/*.pth')        
        else:
            self.files = glob.glob(path+'/*.png')

            if skip_angles:
                new_files = [itm for itm in self.files if ('_90' not in itm and '_270' not in itm)] 
                self.files = new_files


    def __getitem__(self, idx):
        if self.eval_option == 'FID_1':
            this_itm = torch.load(self.files[idx])
            clean_input = (this_itm['input_img'] / 2 + 0.5) * this_itm['input_mask']
            clean_pred = (this_itm['pred_img'] / 2 + 0.5) * this_itm['pred_mask']
            clean_input = (this_itm['input_img'] / 2 + 0.5) * this_itm['input_mask']
            ret = {
                'input': clean_input.squeeze(0), 
                'pred': clean_pred.squeeze(0), 
                }

            return ret
        else:
            filename = self.files[idx]
            img = imread(str(filename)).astype(np.float32)

            img = img.transpose((2, 0, 1))
            img /= 255

            img_t = torch.from_numpy(img).type(torch.FloatTensor)

            return img_t
            
    def __len__(self):
        return len(self.files)


class Tester(object):
    def __init__(self, args):
        self.args = args


    def eval_iou(self):
        self.results_dir = os.path.join('./outputs/inversion_results',self.args.name)
        self.pathnames = sorted(glob.glob(self.results_dir+'/*.pth'))
        iou_ls = []
        
        for i, pathname in enumerate(self.pathnames):
            this_unit = torch.load(pathname)
            input_mask = this_unit['input_mask']
            pred_mask = this_unit['pred_mask']

            overlap_mask = input_mask * pred_mask
            union_mask = input_mask + pred_mask - overlap_mask
            iou = overlap_mask.sum()/union_mask.sum()
            iou_ls.append(iou)

        ious = torch.stack(iou_ls)
        print('Mean IoU:{:4.3f}'.format(ious.mean().item()))
    
    def render_multiview(self, n_views=12):
        # init MeshInversion, which got GAN, mesh template, and renderer
        self.model = MeshInversion(self.args)

        results_dir = os.path.join('./outputs/inversion_results',self.args.name)
        pathnames = sorted(glob.glob(results_dir+'/*.pth'))

        rendering_dir = os.path.join(f'./outputs/multiview_renderings_{n_views}',self.args.name)
        os.makedirs(rendering_dir, exist_ok=True)

        for i, pathname in enumerate(pathnames):
            data = torch.load(pathname)
            idx = data['idx']
               
            pred_tex = data['pred_tex'].cuda()
            pred_shape = data['pred_shape'].cuda()
            scale = torch.tensor([self.args.default_scale]).cuda() 
            translation = data['translation'].cuda() # not in use
            
            for angle in range(0,360,self.args.angle_interval):
                if n_views == 10 and angle in [90, 270]:
                    continue
                if self.args.canonical_pose:
                    original_rot = Rotation.from_euler('xyz', [0, -90, 90-self.args.default_orientation], degrees=True)
                    temp_quat = original_rot.as_quat().astype(np.float32)
                    original_rotation = [temp_quat[-1],temp_quat[0],temp_quat[1],temp_quat[2]]
                else:
                    raise
                rot = Rotation.from_euler('xyz', [0, angle, 0], degrees=True)
                rot_quat = rot.as_quat().astype(np.float32)
                quaternion = [rot_quat[-1],rot_quat[0],rot_quat[1],rot_quat[2]]
                quaternion = quaternion_apply(quaternion, original_rotation)
                img, mask, _ = self.model.render(pred_tex,pred_shape, attn_map=None, rotation=torch.tensor([quaternion]).cuda(), \
                    scale=scale, translation=translation, novel_view=True)
                
                img = img / 2 + 0.5
                img = img + (1-torch.cat([mask,mask,mask],1))
                img = img.squeeze(0)
                img = (img.permute(1, 2, 0)*255).clamp(0, 255).cpu().byte().numpy()
                
                pathname = os.path.join(rendering_dir,f'{idx}_{angle:03d}.png')
                imageio.imwrite(pathname, img)
            
            if i%200 == 0:
                print(f'done {i} out of {len(pathnames)}')

    def eval_fid(self):
        
        inception_model = torch.nn.DataParallel(init_inception()).cuda().eval()

        if self.args.eval_option == 'FID_1':
            path = os.path.join('./outputs/inversion_results',self.args.name)
            data_set = EvalDataset(path=path, eval_option='FID_1',skip_angles=False)
            data_loader = torch.utils.data.DataLoader(data_set, batch_size=40, num_workers=8, \
                        pin_memory=True, drop_last=False, shuffle=False)
            emb_fake = []
            emb_real = []
            for i, data in enumerate(tqdm(data_loader)):
                pred_data = data['pred']
                input_data = data['input']
                emb_fake.append(forward_inception_batch(inception_model, pred_data.cuda()))
                emb_real.append(forward_inception_batch(inception_model, input_data.cuda()))
            
            emb_fake = np.concatenate(emb_fake, axis=0)
            emb_real = np.concatenate(emb_real, axis=0)
            m1, s1 = calculate_stats(emb_fake)
            m2, s2 = calculate_stats(emb_real)
        else:
            if self.args.eval_option == 'FID_12':
                path = os.path.join('./outputs/multiview_renderings_12',self.args.name)
                data_set = EvalDataset(path=path, eval_option='FID_12',skip_angles=False)
            else:
                # reuse the 12-view renderings with skip_angles=True
                path = os.path.join('./outputs/multiview_renderings_12',self.args.name)
                data_set = EvalDataset(path=path, eval_option='FID_10',skip_angles=True)
            
            data_loader = torch.utils.data.DataLoader(data_set, batch_size=40, num_workers=8, \
                        pin_memory=True, drop_last=False, shuffle=False)
            
            # load_gt_stats
            filepath = os.path.join('./datasets/cub', 'cache', 'precomputed_fid_299x299_testval.npz')
            stats = np.load(filepath)
            m2 = stats['stats_m']
            s2 = stats['stats_s'] + np.triu(stats['stats_s'].T, 1)
            
            emb_fake = []
    
            for i, data in enumerate(tqdm(data_loader)):
                emb_fake.append(forward_inception_batch(inception_model, data.cuda()))
            
            emb_fake = np.concatenate(emb_fake, axis=0)
            m1, s1 = calculate_stats(emb_fake)
            fid = calculate_frechet_distance(m1, s1, m2, s2)


        fid = calculate_frechet_distance(m1, s1, m2, s2)

        print('{}:{:.02f}'.format(self.args.eval_option,fid))


if __name__ == "__main__":
    args = Arguments(stage='evaluation').parser().parse_args()

    tester = Tester(args)
    if args.eval_option == 'IoU':
        tester.eval_iou()
    if args.eval_option == 'FID_1':
        tester.eval_fid()
    if args.eval_option == 'FID_12':
        tester.render_multiview(n_views=12)
        tester.eval_fid()
    if args.eval_option == 'FID_10':
        # reuse the 12-view renderings with skip_angles=True, can comment out if FID_12 called before
        # tester.render_multiview(n_views=12)
        tester.eval_fid()

    
