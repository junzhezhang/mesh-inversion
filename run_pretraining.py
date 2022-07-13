import os
# Workaround for PyTorch spawning too many threads
os.environ['OMP_NUM_THREADS'] = '1'

import os.path
import pathlib
from lib.arguments import Arguments
import sys
import time
import math
import glob

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision

from lib.rendering.mesh_template import MeshTemplate
from lib.rendering.utils import qrot
from scipy.spatial.transform import Rotation
from lib.utils.cam_pose import quaternion_apply

from lib.utils.fid import calculate_stats, calculate_frechet_distance, init_inception, forward_inception_batch
from lib.utils.losses import GANLoss, loss_flat

from lib.data.pseudo_dataset import PseudoDatasetForEvaluation

from lib.models.gan import MultiScaleDiscriminator, Generator
from lib.models import cyclegan_networks

from lib.rendering.renderer import Renderer
import scipy

from tqdm import tqdm

import pickle
import torch.nn.functional as F


class ModelWrapper(nn.Module):
    def __init__(self, args):

        super().__init__()
        self.args =  args

        self.generator = Generator(self.args, self.args.latent_dim, symmetric=self.args.symmetric_g, mesh_head=self.args.use_mesh)
        self.generator_running_avg = Generator(self.args, self.args.latent_dim, symmetric=self.args.symmetric_g, mesh_head=self.args.use_mesh)
        self.generator_running_avg.load_state_dict(self.generator.state_dict()) # Same initial weights
        for p in self.generator_running_avg.parameters():
            p.requires_grad = False

        self.discriminator = MultiScaleDiscriminator(self.args, 4) if not self.args.evaluate else None

        if self.args.conditional_text:
            raise NotImplementedError

        self.criterion_gan_uv = GANLoss(self.args.loss, tensor=torch.cuda.FloatTensor).cuda()

        # image space discrimination
        self.mesh_template = MeshTemplate(self.args.mesh_path, is_symmetric=self.args.symmetric_g, multi_gpu=True)
        self.renderer = Renderer(self.args.render_res, self.args.render_res)
        # load cam pose for random view rendering
        filename = os.path.join(self.args.data_dir, 'cache', 'cam_pose_train.pickle')
        with open(filename, 'rb') as filehandler2:
            self.cam_pose = pickle.load(filehandler2)
        ### 2d gan loss
        self.criterion_gan_2d = cyclegan_networks.GANLoss(args.gan_mode, apply_mask=self.args.patchGAN_apply_mask).cuda() 
        self.patchD = cyclegan_networks.define_D(args.input_nc, args.ndf, args.netD,
                                            args.n_layers_D, args.norm, args.init_type, args.init_gain, gpu_ids=None)

    
    def render(self, pred_tex, pred_mesh_map, attn_map=None, others=None,scale=None,translation=None,rotation=None,novel_view=False):
        tic = time.time()
        ### prepare cam pose
        idx_list = others['idx'].detach().cpu().numpy().tolist()
        if scale is None:
            self.scale_ls = [self.cam_pose[idx]['scale'] for idx in idx_list]
            scale = torch.cat(self.scale_ls,0).cuda()
        if translation is None:
            self.translation_ls = [self.cam_pose[idx]['translation'] for idx in idx_list]
            translation = torch.cat(self.translation_ls,0).cuda()
        if rotation is None:
            self.rotation_ls = [self.cam_pose[idx]['rotation'] for idx in idx_list]
            rotation = torch.cat(self.rotation_ls,0).cuda()

        vtx = self.mesh_template.get_vertex_positions(pred_mesh_map) # (B, 482, 3)


        if novel_view:
            ### normalize and do not translate
            center = vtx.mean(dim=-2) # [B,3]
            vtx = vtx - center.unsqueeze(1) #[B,482,3]
            vtx = qrot(rotation, scale.unsqueeze(-1)*vtx)
        else:
            vtx = qrot(rotation, scale.unsqueeze(-1)*vtx) + translation.unsqueeze(1)
        
        vtx = vtx * torch.Tensor([1, -1, -1]).to(vtx.device)

        image_pred, alpha_pred = \
            self.mesh_template.forward_renderer(self.renderer, vtx, pred_tex, \
            num_gpus=1, return_hardmask=True) 

        img = image_pred.permute([0,3,1,2])
        mask = alpha_pred.permute([0,3,1,2])
       
        
        img = img.contiguous()
        mask = mask.contiguous()
        return img, mask
        

    def forward(self, mode, X_tex=None, X_alpha=None, X_mesh=None, C=None, caption=None, others=None, noise=None):

        assert mode in ['inference', 'g+2d', 'd+2d']
        if noise is None:
            noise = torch.randn((X_alpha.shape[0], self.args.latent_dim), device=X_alpha.device)
        
        
        if self.args.num_discriminators == 2 and self.args.texture_resolution >= 512:
            d_weight = [2, 1] # Texture discriminator has a larger weight on the loss
        else:
            d_weight = None # Unweighted
            
        if mode == 'g+2d':
            pred_tex, pred_mesh = self.generator(noise, C, caption, others=others)
            X_fake = torch.cat((pred_tex * X_alpha, X_alpha), dim=1) # Mask output
            X_fake_mesh = pred_mesh

            discriminated, mask = self.discriminator(X_fake, X_fake_mesh, C, caption, others=others)
            loss_gan_uv = self.criterion_gan_uv(discriminated, True, for_discriminator=False, mask=mask, weight=d_weight)
            
            pred_img, pred_mask = self.render(pred_tex, pred_mesh, others=others)
            discriminated_2d = self.patchD(pred_img) # (B, 1, 30, 30)
            loss_gan_2d = self.criterion_gan_2d(discriminated_2d, True, mask=pred_mask)
           
            return loss_gan_uv, loss_gan_2d.unsqueeze(0), pred_tex, pred_mesh
        
        if mode == 'd+2d':
            with torch.no_grad():
                pred_tex, pred_mesh = self.generator(noise, C, caption, others)
                pred_img, pred_mask = self.render(pred_tex, pred_mesh, others=others)

                X_fake = torch.cat((pred_tex * X_alpha, X_alpha), dim=1) # Mask output
                X_fake_mesh = pred_mesh
                    
                X_real = torch.cat((X_tex, X_alpha), dim=1)
                assert (X_mesh is None) == (pred_mesh is None)
                X_combined = torch.cat((X_fake, X_real), dim=0)
                C_combined = torch.cat((C, C), dim=0) if C is not None else None
                others_combined = {key: torch.cat((x, x), dim=0) for key, x in others.items()} if others is not None else None
                caption_combined = [torch.cat((x, x), dim=0) for x in caption] if caption is not None else None
                if pred_mesh is not None:
                    X_real_mesh = X_mesh
                    X_combined_mesh = torch.cat((X_fake_mesh, X_real_mesh), dim=0)
                else:
                    X_combined_mesh = None
            discriminated, mask = self.discriminator(X_combined, X_combined_mesh, C_combined, caption_combined, others_combined)
            discriminated_fake, discriminated_real = self.divide_pred(discriminated)
            mask_fake, mask_real = self.divide_pred(mask)
            loss_fake = self.criterion_gan_uv(discriminated_fake, False, for_discriminator=True, mask=mask_fake, weight=d_weight)
            loss_real = self.criterion_gan_uv(discriminated_real, True, for_discriminator=True, mask=mask_real, weight=d_weight)

            discriminated_2d_fake = self.patchD(pred_img)
            discriminated_2d_real = self.patchD(others['image'])
            loss_2d_fake = self.criterion_gan_2d(discriminated_2d_fake, False, mask=pred_mask)
            loss_2d_real = self.criterion_gan_2d(discriminated_2d_real, True, mask=others['mask'])

            return loss_fake, loss_real, pred_tex, pred_mesh, loss_2d_fake, loss_2d_real
        
        if mode == 'inference':
            with torch.no_grad():
                pred_tex, pred_mesh = self.generator(noise, C, caption, others=others)
                attn_map = None
            return pred_tex, pred_mesh, attn_map


    def divide_pred(self, pred):
        if pred is None:
            return None, None
        
        if type(pred) == list:
            fake = [x[:x.shape[0]//2] if x is not None else None for x in pred]
            real = [x[x.shape[0]//2:] if x is not None else None for x in pred]
        else:
            fake = pred[:pred.shape[0]//2]
            real = pred[pred.shape[0]//2:]

        return fake, real

class GANTrainer(object):

    def __init__(self, args):
        self.args = args

        self.data_dir = args.data_dir
        if self.args.export_sample:
            self.args.evaluate = True
        self.args.use_mesh = not self.args.texture_only
        # A few safety checks...
        if self.args.num_discriminators >= 3:
            assert self.args.texture_resolution >= 512
                            
        if self.args.norm_g == 'syncbatch':
            # Import library for synchronized batch normalization
            from lib.sync_batchnorm import DataParallelWithCallback

        
        ### copy to model wraper from here
        if self.args.tensorboard and not self.args.evaluate:
            import shutil
            from torch.utils.tensorboard import SummaryWriter
            log_dir = 'tensorboard_pretrain/' + self.args.name
            if not self.args.continue_train:
                    shutil.rmtree(log_dir, ignore_errors=True) # Delete logs
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

        ### define dataset and dataloader
        if self.args.dataset == 'cub':
            from lib.data.cub_pseudo_dataset import CubPseudoDataset
            self.train_ds = CubPseudoDataset(self.args)
        else:
            raise ValueError('Invalid dataset')

        if self.args.mesh_path == 'autodetect':
            self.args.mesh_path = self.train_ds.suggest_mesh_template()
            print('Using autodetected mesh', self.args.mesh_path)
                
        if self.args.num_discriminators == -1:
            # Autodetect
            self.args.num_discriminators = self.train_ds.suggest_num_discriminators()
                
        if self.args.truncation_sigma < 0:
            # Autodetect
            self.args.truncation_sigma = self.train_ds.suggest_truncation_sigma()
            print(f'Using truncation sigma {self.args.truncation_sigma} for evaluation')
            
        self.eval_ds = PseudoDatasetForEvaluation(self.train_ds)

        self.train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                                pin_memory=True, drop_last=True, shuffle=True)
            
        self.eval_loader = torch.utils.data.DataLoader(self.eval_ds, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                                pin_memory=True, shuffle=False)

        ### define mesh template and render
        self.mesh_template = MeshTemplate(self.args.mesh_path, is_symmetric=self.args.symmetric_g)
        # For real-time FID evaluation
        if not self.args.export_sample:
            evaluation_res = 299 # Same as Inception input resolution
        else:
            evaluation_res = 512 # For exporting images: higher resolution
        renderer = Renderer(evaluation_res, evaluation_res)
        # NOTE: relax gpu_ids
        # renderer = nn.DataParallel(renderer, gpu_ids)  
        self.renderer = nn.DataParallel(renderer)  

        if not args.export_sample:
            ### define fid model and load stats
            self.inception_model = nn.DataParallel(init_inception()).cuda().eval()
            # Statistics for real images are computed only once and cached
            self.m_real_train, self.s_real_train = None, None
            self.m_real_val, self.s_real_val = None, None

            # Load precomputed statistics to speed up FID computation
            stats = np.load(os.path.join(self.data_dir, 'cache', f'precomputed_fid_{evaluation_res}x{evaluation_res}_train.npz'), allow_pickle=True)
            self.m_real_train = stats['stats_m']
            self.s_real_train = stats['stats_s'] + np.triu(stats['stats_s'].T, 1)
            assert stats['num_images'] == len(self.train_ds), 'Number of images does not match'
            assert stats['resolution'] == evaluation_res, 'Resolution does not match'
            stats = None

            if self.args.dataset == 'cub':
                stats = np.load(os.path.join(self.data_dir, 'cache', f'precomputed_fid_{evaluation_res}x{evaluation_res}_testval.npz'), allow_pickle=True)
                self.m_real_val = stats['stats_m']
                self.s_real_val = stats['stats_s'] + np.triu(stats['stats_s'].T, 1)
                self.n_images_val = stats['num_images']
                assert self.n_images_val <= len(self.train_ds), 'Not supported'
                assert stats['resolution'] == evaluation_res, 'Resolution does not match'
                stats = None
        
        ### define and load models
        # define models, load state, parallel
        if self.args.norm_g == 'syncbatch':
            dataparallel = DataParallelWithCallback
            print('Using SyncBN')
        else:
            dataparallel = nn.DataParallel

        self.model = dataparallel(ModelWrapper(self.args).cuda())

        ### define optimizers
        self.generator = self.model.module.generator
        self.generator_running_avg = self.model.module.generator_running_avg
        self.discriminator = self.model.module.discriminator
        self.patchD = self.model.module.patchD

        if not self.args.evaluate:
            g_parameters = self.generator.parameters()
            d_parameters = self.discriminator.parameters()
            patchd_parameters = self.patchD.parameters()
   
        if not self.args.evaluate:
            self.optimizer_g = optim.Adam(g_parameters, lr=self.args.lr_g, betas=(0, 0.9))
            self.optimizer_d = optim.Adam(d_parameters, lr=self.args.lr_d, betas=(0, 0.9))
            self.optimizer_patchd = optim.Adam(patchd_parameters, lr=self.args.lr_d, betas=(0, 0.9))


        ### tensorboard & load models
        self.d_fake_curve = [0]
        self.d_real_curve = [0]
        self.g_curve = [0]
        self.flat_curve = [0]
        self.total_it = 0
        self.epoch = 0

        self.checkpoint_dir = 'checkpoints_gan/' + self.args.name
        if self.args.continue_train or self.args.evaluate:
            # Load last checkpoint
            if self.args.which_epoch == 'best':
                which_epoch = 'latest' # Bypass (the search will be done later)
            else:
                which_epoch = self.args.which_epoch
            chk = torch.load(os.path.join(self.checkpoint_dir, f'checkpoint_{which_epoch}.pth'),
                            map_location=lambda storage, loc: storage)
            if 'epoch' in chk:
                self.epoch = chk['epoch']
                self.total_it = chk['iteration']
                self.g_curve = chk['g_curve']
                self.d_fake_curve = chk['d_fake_curve']
                self.d_real_curve = chk['d_real_curve']
                self.flat_curve = chk['flat_curve']
                self.generator.load_state_dict(chk['generator'])
            self.generator_running_avg.load_state_dict(chk['generator_running_avg'])
            if self.args.continue_train:
                self.optimizer_g.load_state_dict(chk['self.optimizer_g'])
                self.discriminator.load_state_dict(chk['discriminator'])
                self.optimizer_d.load_state_dict(chk['self.optimizer_d'])
                
                self.optimizer_patchd.load_state_dict(chk['optimizer_patchd'])
                self.patchD.load_state_dict(chk['patchD'])

                print(f'Resuming from epoch {self.epoch}')
            else:
                if 'epoch' in chk:
                    print(f'Evaluating epoch {self.epoch}')
                self.args.epochs = -1 # Disable training
            chk = None
            
        if not self.args.evaluate:
            pathlib.Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            self.log_file = open(os.path.join(self.checkpoint_dir, 'log.txt'), 'a' if self.args.continue_train else 'w', buffering=1) # Line buffering
            print(' '.join(sys.argv), file=self.log_file)
        else:
            self.log_file = None



    def save_checkpoint(self,it):
        out_dict = {
            'self.optimizer_g': self.optimizer_g.state_dict(),
            'self.optimizer_d': self.optimizer_d.state_dict(),
            'generator': self.generator.state_dict(),
            'generator_running_avg': self.generator_running_avg.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'epoch': self.epoch,
            'iteration': self.total_it,
            'g_curve': self.g_curve,
            'd_fake_curve': self.d_fake_curve,
            'd_real_curve': self.d_real_curve,
            'flat_curve': self.flat_curve,
            'self.args': vars(self.args),
        }
        out_dict['patchD'] = self.patchD.state_dict()
        out_dict['optimizer_patchd'] = self.optimizer_patchd.state_dict()

        torch.save(out_dict, os.path.join(self.checkpoint_dir, f'checkpoint_{it}.pth'))


    def train(self):
        while self.epoch < self.args.epochs:

            start_time = time.time()
            for i, data in enumerate(self.train_loader):

                X_tex = data['texture'].cuda()
                X_alpha = data['texture_alpha'].cuda()
                
                if self.args.conditional_class:
                    C = data['class'].cuda()
                    caption = None
                    others = None
                else:
                    C, caption, others = None, None, None
                    
                if self.args.use_mesh:
                    X_mesh = data['mesh'].cuda()
                else:
                    X_mesh = None
                others = {}
                others['idx'] = data['idx']
                
                if 'image_256' in data:
                    others['image'] = data['image_256']
                else:
                    others['image'] = F.interpolate(data['image'],(256,256)).cuda()
                others['mask'] = (others['image'] == 0.0).int().type(torch.float32)[:,0:1,:,:]                  
                    
                if self.total_it % (1 + self.args.d_steps_per_g) == 0:
                    ### G
                    self.optimizer_g.zero_grad()
                    loss_gan_uv, loss_gan_2d, pred_tex, pred_mesh = self.model('g+2d', None, X_alpha, None, C, caption, others)

                    if self.args.use_mesh:
                        vtx = self.mesh_template.get_vertex_positions(pred_mesh)
                        flat_loss = loss_flat(self.mesh_template.mesh, self.mesh_template.compute_normals(vtx))
                        self.flat_curve.append(flat_loss.item())
                    else:
                        flat_loss = 0
                    
                    loss = loss_gan_uv.mean() + loss_gan_2d.mean() * self.args.img_gan_loss_wt + self.args.mesh_regularization*flat_loss
                    # import 
                    loss.backward()


                    self.optimizer_g.step()
                    self.update_generator_running_avg(self.epoch)
                    self.g_curve.append(loss_gan_uv.mean().item()) 

                    if self.args.tensorboard:
                        self.writer.add_scalar(f'gan_{self.args.loss}/g_uv', loss_gan_uv.mean().item(), self.total_it)
                        self.writer.add_scalar(f'gan_{self.args.loss}/g_2d', loss_gan_2d.mean().item(), self.total_it)
                        if self.args.use_mesh:
                            self.writer.add_scalar('flat', flat_loss.item(), self.total_it)

                else:
                    ### Ds, both UG D and patchD
                    self.optimizer_d.zero_grad()
                    loss_fake, loss_real, pred_tex, pred_mesh, loss_2d_fake, loss_2d_real = self.model('d+2d', X_tex, X_alpha, X_mesh, C, caption, others)
                    loss_fake = loss_fake.mean()
                    loss_real = loss_real.mean()
                    loss = loss_fake + loss_real
                    loss.backward()
                    self.optimizer_d.step()
                    ### patchD loop
                    self.optimizer_d.zero_grad()
                    loss_2d = loss_2d_fake.mean() + loss_2d_real.mean()
                    loss_2d.backward()
                    self.optimizer_patchd.step()
                    
                    ### log
                    self.d_fake_curve.append(loss_fake.item())
                    self.d_real_curve.append(loss_real.item())

                    if self.args.tensorboard:
                        self.writer.add_scalar(f'gan_{self.args.loss}/d_fake_loss', loss_fake.item(), self.total_it)
                        self.writer.add_scalar(f'gan_{self.args.loss}/d_real_loss', loss_real.item(), self.total_it)
                        self.writer.add_scalar(f'gan_{self.args.loss}/patchd_fake_loss', loss_2d_fake.mean().item(), self.total_it)
                        self.writer.add_scalar(f'gan_{self.args.loss}/patchd_real_loss', loss_2d_real.mean().item(), self.total_it)                        


           
                if self.total_it % 50 == 0:
                    self.log('[{}] epoch {}, {}/{}, g_loss {:.5f} d_fake_loss {:.5f} d_real_loss {:.5f} flat {:.5f}'.format(
                                                                            self.total_it, self.epoch, i, len(self.train_loader),
                                                                            self.g_curve[-1], self.d_fake_curve[-1], self.d_real_curve[-1],
                                                                            self.flat_curve[-1]))
                    tic = time.time()
                    self.writer.flush()
                    toc = time.time()
                self.total_it += 1

            self.epoch += 1
            print('epoch',self.epoch)
            
            self.log('Time per epoch: {:.3f} s'.format(time.time() - start_time))
            
            # LR
            if self.epoch >= self.args.lr_decay_after and self.epoch < self.args.epochs:
                factor = 1 - min(max((self.epoch - self.args.lr_decay_after)/(self.args.epochs - self.args.lr_decay_after), 0), 1)
                for param_group in self.optimizer_g.param_groups:
                    param_group['lr'] = self.args.lr_g * factor
                for param_group in self.optimizer_d.param_groups:
                    param_group['lr'] = self.args.lr_d * factor
                            
            if self.epoch % self.args.save_freq == 0:
                self.save_checkpoint('latest')
            if self.epoch % self.args.checkpoint_freq == 0:
                self.save_checkpoint(str(self.epoch))
            if self.epoch % self.args.evaluate_freq == 0 and not self.args.texture_only:
                self.evaluate_fid(self.writer, self.total_it, data['idx'])

    
    def evaluate(self):
        raise
    
    def export_sample(self):
        print(f'Exporting sample of {args.batch_size} objects')
        class_ls = []
        with torch.no_grad():
            indices = np.random.choice(len(self.train_ds), size=self.args.batch_size, replace=False)
            if self.args.conditional_class:
                c = torch.LongTensor([self.train_ds.classes[i] for i in indices]).cuda()
                caption = None

            else:
                c, caption = None, None

            noise = torch.randn(self.args.batch_size, self.args.latent_dim)

            # Gaussian truncation trick
            sigma = self.args.truncation_sigma
            while (noise.abs() > sigma).any():
                # Rejection sampling
                mask = noise.abs() > sigma
                noise[mask] = torch.randn_like(noise[mask])

            self.generator_running_avg.eval()
            noise = noise.cuda()
            pred_tex, pred_mesh_map, attn_map = self.model('inference', None, None, C=c, caption=caption, noise=noise)


            vtx = self.mesh_template.get_vertex_positions(pred_mesh_map)
            vtx_obj = vtx.clone()
            vtx_obj[..., :] = vtx_obj[..., [0, 2, 1]] # Swap Y and Z (the result is Y up)
            output_dir = os.path.join('output', self.args.name)
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            for i, v in enumerate(vtx_obj):
                self.mesh_template.export_obj(os.path.join(output_dir, f'mesh_{i}'), v, pred_tex[i]/2 + 0.5)
            
            
            rotation = self.train_ds.data['rotation'][indices].cuda()
            scale = self.train_ds.data['scale'][indices].cuda()
            translation = self.train_ds.data['translation'][indices].cuda()
            
            vtx = qrot(rotation, scale.unsqueeze(-1)*vtx) + translation.unsqueeze(1)
            vtx = vtx * torch.Tensor([1, -1, -1]).to(vtx.device)

            image_pred, alpha_pred = self.mesh_template.forward_renderer(self.renderer, vtx, pred_tex,
                                                                    num_gpus=len(gpu_ids),
                                                                    return_hardmask=True)
            
            image_pred[alpha_pred.expand_as(image_pred) == 0] = 1
            image_pred = image_pred.permute(0, 3, 1, 2)/2 + 0.5
            image_pred = F.avg_pool2d(image_pred, 2) # Anti-aliasing

            import imageio
            import torchvision
            image_grid = torchvision.utils.make_grid(image_pred, nrow=8, padding=0)
            image_grid = (image_grid.permute(1, 2, 0)*255).clamp(0, 255).cpu().byte().numpy()

            imageio.imwrite(f'output/{self.args.name}.png', image_grid)


    def log(self,text):
        if self.log_file is not None:
            print(text, file=self.log_file)
        print(text)

    def to_grid_tex(self, x):
        with torch.no_grad():
            return torchvision.utils.make_grid((x.data[:, :3]+1)/2, nrow=4)

    def to_grid_mesh(self, x):
        with torch.no_grad():
            x = x.data[:, :3]
            minv = x.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]
            maxv = x.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
            x = (x - minv)/(maxv-minv)
            return torchvision.utils.make_grid(x, nrow=4)

    def update_generator_running_avg(self, epoch):
        with torch.no_grad():
            # This heuristic does not affect the final result, it is just done for visualization purposes.
            # If alpha is very high (e.g. 0.999) it may take a while to visualize correct results on TensorBoard,
            # (or estimate reliable FID scores), therefore we lower alpha for the first few epochs.
            if epoch < 10:
                alpha = math.pow(self.args.g_running_average_alpha, 100)
            elif epoch < 100:
                alpha = math.pow(self.args.g_running_average_alpha, 10)
            else:
                alpha = self.args.g_running_average_alpha
            g_state_dict = self.generator.state_dict()
            for k, param in self.generator_running_avg.state_dict().items():
                if torch.is_floating_point(param):
                    param.mul_(alpha).add_(g_state_dict[k], alpha=1-alpha)
                else:
                    param.fill_(g_state_dict[k])


    def evaluate_fid(self, writer=None, it=None, visualization_indices=None, fast=False):

        emb_arr_fake_combined = []
        emb_arr_fake_texture_only = []
        emb_arr_fake_mesh_only = []
        emb_arr_real = []

        # Grid for visualization
        if visualization_indices is not None:
            indices_to_render = visualization_indices.numpy()
            shuffle_idx = np.argsort(np.argsort(indices_to_render)) # To restore the original order
        else:
            indices_to_render = np.random.choice(len(self.train_ds), size=16, replace=False)
            shuffle_idx = None
            
        with torch.no_grad():
            self.generator_running_avg.eval()

            sample_real = []
            sample_fake = []
            sample_fake_texture_only = []
            sample_fake_mesh_only = []
            sample_text = [] # For models trained with captions
            sample_tex_real = []
            sample_tex_fake = []
            sample_mesh_map_fake = []
            
            if self.args.evaluate:
                # Deterministic seed, but only in evaluation mode since we do not want to reset
                # the random state while we train the model (it would cripple the model).
                # Note that FID scores might still exhibit some variability depending on the batch size.
                torch.manual_seed(1234)
            
            it_cnt = 0

            for data in tqdm(self.eval_loader):
                for k in ['texture', 'mesh', 'translation', 'scale', 'rotation']:
                    if k in data:
                        data[k] = data[k].cuda()

                has_pseudogt = 'texture' in data and not fast
                # print('has_pseudogt',has_pseudogt)

                if self.m_real_train is None:
                    # Compute real (only if not cached)
                    assert 'image' in data
                    assert data['image'].shape[2] == evaluation_res
                    assert data['image'].shape[3] == evaluation_res
                    emb_arr_real.append(forward_inception_batch(self.inception_model, data['image'].cuda()))

                if self.args.conditional_class:
                    c = data['class'].cuda()
                    caption = None
                    others = None
                else:
                    c, caption = None, None

                noise = torch.randn(data['idx'].shape[0], self.args.latent_dim)
                
                # Gaussian truncation trick
                sigma = self.args.truncation_sigma
                while (noise.abs() > sigma).any():
                    # Rejection sampling
                    mask = noise.abs() > sigma
                    noise[mask] = torch.randn_like(noise[mask])

                noise = noise.cuda()
                
                if noise.shape[0] % len(gpu_ids) == 0:
                    pred_tex, pred_mesh_map, attn_map = self.model('inference', None, None, C=c, caption=caption, others=others, noise=noise)

                else:
                    # Batch dimension is not divisible by number of GPUs --> pad
                    original_bsz = noise.shape[0]
                    padding_bsz = len(gpu_ids) - (noise.shape[0] % len(gpu_ids))
                    def pad_batch(batch):
                        return torch.cat((batch, torch.zeros((padding_bsz, *batch.shape[1:]),
                                                            dtype=batch.dtype).to(batch.device)), dim=0)
                        
                    noise_pad = pad_batch(noise)
                    if c is not None:
                        c_pad = pad_batch(c)
                    else:
                        c_pad = None
                    if caption is not None:
                        caption_pad = tuple([pad_batch(x) for x in caption])
                    else:
                        caption_pad = None
                    if others is not None:
                        others_pad = {key: pad_batch(itm) for key, itm in others.items()}
                    else:
                        others_pad = None
                    pred_tex, pred_mesh_map, attn_map = self.model('inference', None, None, C=c_pad, caption=caption_pad, others=others_pad, noise=noise_pad)
                    
                    # Unpad
                    pred_tex = pred_tex[:original_bsz]
                    pred_mesh_map = pred_mesh_map[:original_bsz]
                    if attn_map is not None:
                        attn_map = attn_map[:original_bsz]
                
                def render_and_score(input_mesh_map, input_texture, output_array):
                    vtx = self.mesh_template.get_vertex_positions(input_mesh_map)
                    vtx = qrot(data['rotation'], data['scale'].unsqueeze(-1)*vtx) + data['translation'].unsqueeze(1)
                    vtx = vtx * torch.Tensor([1, -1, -1]).to(vtx.device)

                    image_pred, _ = self.mesh_template.forward_renderer(self.renderer, vtx, input_texture, len(gpu_ids))
                    image_pred = image_pred.permute(0, 3, 1, 2)/2 + 0.5
                    
                    emb = forward_inception_batch(self.inception_model, image_pred)
                    output_array.append(emb)
                    return image_pred # Return images for visualization

                out_combined = render_and_score(pred_mesh_map, pred_tex, emb_arr_fake_combined)
                
                mask, = np.where(np.isin(data['idx'].cpu().numpy(), indices_to_render))
                if len(mask) > 0:
                    sample_fake.append(out_combined[mask].cpu())
                    sample_mesh_map_fake.append(pred_mesh_map[mask].cpu())
                    sample_tex_fake.append(pred_tex[mask].cpu())
                    if has_pseudogt:
                        sample_real.append(data['image'][mask])
                        sample_tex_real.append(data['texture'][mask].cpu())

                    
                if has_pseudogt:
                    out_combined = render_and_score(data['mesh'], pred_tex, emb_arr_fake_texture_only)
                    if len(mask) > 0:
                        sample_fake_texture_only.append(out_combined[mask].cpu())
                    out_combined = render_and_score(pred_mesh_map, data['texture'], emb_arr_fake_mesh_only)
                    if len(mask) > 0:
                        sample_fake_mesh_only.append(out_combined[mask].cpu())
        
        emb_arr_fake_combined = np.concatenate(emb_arr_fake_combined, axis=0)
        if has_pseudogt:
            emb_arr_fake_texture_only = np.concatenate(emb_arr_fake_texture_only, axis=0)
            emb_arr_fake_mesh_only = np.concatenate(emb_arr_fake_mesh_only, axis=0)
            sample_real = torch.cat(sample_real, dim=0)
        sample_fake = torch.cat(sample_fake, dim=0)
        sample_mesh_map_fake = torch.cat(sample_mesh_map_fake, dim=0)
        sample_tex_fake = torch.cat(sample_tex_fake, dim=0)
        if has_pseudogt:
            sample_fake_texture_only = torch.cat(sample_fake_texture_only, dim=0)
            sample_fake_mesh_only = torch.cat(sample_fake_mesh_only, dim=0)
            sample_tex_real = torch.cat(sample_tex_real, dim=0)
        if shuffle_idx is not None:
            sample_fake = sample_fake[shuffle_idx]
            sample_mesh_map_fake = sample_mesh_map_fake[shuffle_idx]
            sample_tex_fake = sample_tex_fake[shuffle_idx]
            if has_pseudogt:
                sample_real = sample_real[shuffle_idx]
                sample_fake_texture_only = sample_fake_texture_only[shuffle_idx]
                sample_fake_mesh_only = sample_fake_mesh_only[shuffle_idx]
                sample_tex_real = sample_tex_real[shuffle_idx]
            
        if self.m_real_train is None:
            emb_arr_real = np.concatenate(emb_arr_real, axis=0)
            self.m_real_train, self.s_real_train = calculate_stats(emb_arr_real)

        m1, s1 = calculate_stats(emb_arr_fake_combined)
        fid = calculate_frechet_distance(m1, s1, self.m_real_train, self.s_real_train)
        self.log('FID (training set): {:.02f}'.format(fid)) 

        if has_pseudogt:
            m2, s2 = calculate_stats(emb_arr_fake_texture_only)
            fid_texture = calculate_frechet_distance(m2, s2, self.m_real_train, self.s_real_train)
            self.log('Texture-only FID (training set): {:.02f}'.format(fid_texture))

            m3, s3 = calculate_stats(emb_arr_fake_mesh_only)
            fid_mesh = calculate_frechet_distance(m3, s3, self.m_real_train, self.s_real_train)
            self.log('Mesh-only FID (training set): {:.02f}'.format(fid_mesh))
        
        if self.m_real_val is not None and not fast:
            # Make sure the number of images is the same as that of the test set
            if self.args.evaluate:
                np.random.seed(1234)
            val_indices = np.random.choice(len(self.train_ds), size=self.n_images_val, replace=False)
            
            m1_val, s1_val = calculate_stats(emb_arr_fake_combined[val_indices])
            fid_val = calculate_frechet_distance(m1_val, s1_val, self.m_real_val, self.s_real_val)
            self.log('FID (validation set): {:.02f}'.format(fid_val))

            if has_pseudogt:
                m2_val, s2_val = calculate_stats(emb_arr_fake_texture_only[val_indices])
                fid_texture_val = calculate_frechet_distance(m2_val, s2_val, self.m_real_val, self.s_real_val)
                self.log('Texture-only FID (validation set): {:.02f}'.format(fid_texture_val))

                m3_val, s3_val = calculate_stats(emb_arr_fake_mesh_only[val_indices])
                fid_mesh_val = calculate_frechet_distance(m3_val, s3_val, self.m_real_val, self.s_real_val)
                self.log('Mesh-only FID (validation set): {:.02f}'.format(fid_mesh_val))
        
        if self.args.tensorboard and not self.args.evaluate:
            self.writer.add_image('image/real_tex', self.to_grid_tex(sample_tex_real), it)
            self.writer.add_image('image/fake_tex', self.to_grid_tex(sample_tex_fake), it)
            self.writer.add_image('image/fake_mesh', self.to_grid_mesh(sample_mesh_map_fake), it)
            
            grid_fake = torchvision.utils.make_grid(sample_fake, nrow=4)
            grid_fake_texture_only = torchvision.utils.make_grid(sample_fake_texture_only, nrow=4)
            grid_fake_mesh_only = torchvision.utils.make_grid(sample_fake_mesh_only, nrow=4)
            grid_real = torchvision.utils.make_grid(sample_real, nrow=4)
            self.writer.add_image('render/fake', grid_fake, it)
            self.writer.add_image('render/fake_texture', grid_fake_texture_only, it)
            self.writer.add_image('render/fake_mesh', grid_fake_mesh_only, it)
            self.writer.add_image('render/real', grid_real, it)
            
           
            self.writer.add_scalar('fid/combined', fid, it)
            if self.m_real_val is not None:
                self.writer.add_scalar('fid/combined_val', fid_val, it)
            self.writer.add_scalar('fid/texture_only', fid_texture, it)
            self.writer.add_scalar('fid/mesh_only', fid_mesh, it)
            
        return fid
    


if __name__ == "__main__":
    args = Arguments(stage='pretraining').parser().parse_args()
    # self.args.device = torch.device('cuda:'+str(self.args.gpu_ids) if torch.cuda.is_available() else 'cpu')\
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    print('Using {} GPUs: {}'.format(len(gpu_ids), gpu_ids))
    torch.cuda.set_device(min(gpu_ids))
    trainer = GANTrainer(args)
    if args.export_sample:
        trainer.export_sample()
    elif args.evaluate:
        trainer.evaluate_fid()
    else:
        trainer.train()








