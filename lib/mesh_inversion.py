import os

import torch
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable


from lib.external.ChamferDistancePytorch.chamfer_python import distChamfer, distChamfer_downsample
from lib.utils import vgg_feat

from lib.rendering.mesh_template import MeshTemplate
from lib.rendering.utils import qrot
from lib.rendering.renderer import Renderer

from lib.models.gan import Generator

from torch.utils.tensorboard import SummaryWriter
from lib.utils.mask_proj import mask2proj, farthest_point_sample, get_vtx_color, grid_sample_from_vtx

from lib.utils.losses import GANLoss, loss_flat
from lib.utils.losses import GANLoss, loss_flat
from lib.utils.common_utils import LRScheduler

class MeshInversion(object):

    def __init__(self, args):
        ### init seed 1234
        torch.manual_seed(1234)

        self.args = args
        self.stage = None
        
        ### create model & load
        use_mesh = not args.texture_only
        
        self.G = Generator(args, args.latent_dim, symmetric=args.symmetric_g, mesh_head=use_mesh)
        
        chk = torch.load(os.path.join('./checkpoints_gan', args.checkpoint_dir,'checkpoint_latest.pth'),
                     map_location=lambda storage, loc: storage)
        self.G.load_state_dict(chk['generator_running_avg'])
        self.G = self.G.cuda()
        self.G.eval()

        ### resnet transform, borrowed from UMR
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.mesh_template = MeshTemplate(args.mesh_path, is_symmetric=args.symmetric_g)
        ### renderer
        # For real-time FID evaluation
        if self.args.img_size == 256:
            evaluation_res = 256
        else:
            if not args.export_sample:
                evaluation_res = 299 # Same as Inception input resolution
            else:
                evaluation_res = 512 # For exporting images: higher resolution
        renderer = Renderer(evaluation_res, evaluation_res)
        self.renderer = renderer

        ### losses
        args.loss = 'hinge' # NOTE default value for convmesh
        self.criterion_gan = GANLoss(args.loss, tensor=torch.cuda.FloatTensor).cuda()
        if self.args.chamfer_texture_feat_loss: 
            self.vgg_feat = vgg_feat.VGG19().cuda()

        batch_size = 1
        self.z = torch.zeros((batch_size, self.args.latent_dim)).normal_().cuda()
        self.z = Variable(self.z, requires_grad=True)
        self.z_optim = torch.optim.Adam([self.z], lr=self.args.z_lrs[0], betas=(0,0.99))
        self.z_scheduler = LRScheduler(self.z_optim, self.args.warm_up)
        
        self.scale =  torch.zeros((batch_size, 1)).normal_().cuda()
        self.translation = torch.zeros((batch_size, 3)).normal_().cuda() # z dimension is dummy
        self.rotation = torch.zeros((batch_size, 4)).normal_().cuda()

        if self.args.update_pose:
            self.scale =  Variable(self.scale, requires_grad=True)
            self.translation = Variable(self.translation, requires_grad=True)
            self.rotation = Variable(self.rotation, requires_grad=True)
            self.pose_optim = torch.optim.Adam([self.scale,self.translation,self.rotation], lr=self.args.z_lrs[0], betas=(0,0.99))
            self.pose_scheduler = LRScheduler(self.pose_optim, self.args.warm_up)

        # for log the loss
        self.curr_step = -1


    def set_target(self, idx, data, seq):
        """
        assume with batch size, but 1
        make target with range [-1, 1], need mask afterwards
        read pseudo GT for get the alpha, for anti_fake_loss
        idx: that in the training data
        seq: the runing sequence
        """
        self.img_path = data['img_path'][0]
        target = data['img'].type(torch.float32).cuda()
        # [0,1]-> [-1,1]
        self.target = (target - 0.5) * 2
        self.mask_target = data['mask'].type(torch.float32).cuda()
        if len(self.mask_target.shape) == 3:
            self.mask_target.unsqueeze_(0)
        self.idx = idx
        self.seq = seq
        if self.args.conditional_class:
            self.c = data['class'].cuda()

        # cam pose
        batch_size = self.target.shape[0]
        data['sfm_pose'] = data['sfm_pose'].type(torch.float32)
        
        self.scale.data = data['sfm_pose'][:,:1].cuda()
        self.translation.data = torch.cat([data['sfm_pose'][:,1:3], torch.zeros(batch_size,1)],1).cuda()
        self.rotation.data = data['sfm_pose'][:,-4:].cuda()

        # tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join('./tensorboard_inversion',self.args.name,str(self.idx)))
        
        # for chamfer losses
        self.vtx_target, self.color_target =  get_vtx_color(self.mask_target.clone(), self.target.clone())

        
    def downsample(self, dense_pcd, n=2048):
 
        idx = farthest_point_sample(dense_pcd,n)
        sparse_pcd = dense_pcd[0,idx]
        return sparse_pcd

    def prepare_input(self,batch_size=1):
        """
        prepare class and caption
        """
        #### forward
        if self.args.dataset == 'p3d' and self.args.conditional_class:
            c = self.car_class(self.idx)
            caption = None
            others = None
            return c, caption, others

        if self.args.conditional_class or self.args.conditional_text:
            if self.args.conditional_class:
                if batch_size >  1:
                    c = torch.cat([self.c]*batch_size,0)
                else: 
                    c = self.c
                caption = None
            elif self.args.conditional_text:
                raise
                c = None
                caps = []
                cap_lengths = []
                for i in indices:
                    cap, cap_length = train_ds.get_random_caption(i)
                    caps.append(cap)
                    cap_lengths.append(cap_length)
                caption = (torch.LongTensor(caps).cuda(), torch.LongTensor(cap_lengths).cuda())
        else:
            c, caption = None, None
        
        if self.args.conditional_encoding:
            raise
        else:
            others = None

        return c, caption, others

    def render(self, pred_tex, pred_mesh_map, attn_map, scale=None, translation=None, rotation=None, novel_view=False, given_vtx_3d=None):
        """
        all rendering goes here
        default is using the extrinsics from the target image, unless specified.
        """
        
        batch_size = pred_tex.shape[0]

        vtx = self.mesh_template.get_vertex_positions(pred_mesh_map) # (B, 482, 3)

        vtx_color = self.mesh_template.get_vertex_colors(pred_tex) # (B, 482, 3)

        batch_size = vtx.shape[0]
        if scale is None:
            if batch_size > 1:
                scale = self.scale.repeat(batch_size,1)
                translation = self.translation.repeat(batch_size,1)
                rotation = self.rotation.repeat(batch_size,1)
            else:
                scale = self.scale
                translation = self.translation
                rotation = self.rotation

        if novel_view:
            ### normalize and do not translate
            center = vtx.mean(dim=-2) # [B,3]
            vtx = vtx - center.unsqueeze(1) #[B,482,3]
            vtx = qrot(rotation, scale.unsqueeze(-1)*vtx)
        else:
            vtx = qrot(rotation, scale.unsqueeze(-1)*vtx) + translation.unsqueeze(1)
        
        vtx = vtx * torch.Tensor([1, -1, -1]).to(vtx.device)

        frontal_mask = self.mesh_template.get_frontal_vertex_indices(vtx)
        
        if given_vtx_3d is not None:
            vtx = given_vtx_3d

        # NOTE: not all used for computation, some are parsed for visual purposes
        xy = vtx[:,:,:2]
        others = {
            'vtx': xy,
            'vtx_color': vtx_color,
            'frontal_mask': frontal_mask,
            'vtx_3d': vtx
        }

        image_pred, alpha_pred = \
            self.mesh_template.forward_renderer(self.renderer, vtx, pred_tex, \
            num_gpus=len(self.args.gpu_ids), return_hardmask=True) 

        img = image_pred.permute([0,3,1,2])
        mask = alpha_pred.permute([0,3,1,2])
       
        img = img.contiguous()
        mask = mask.contiguous()
        return img, mask, others


    def criterion(self, image_pred, mask_pred, pred_tex=None, pred_mesh_map=None, c=None, \
        caption=None, stage='train', others=None):
        """
        compute losses
        """        
        self.stage = stage

        if self.target.shape[0] < image_pred.shape[0]:
            target  = self.target.repeat(image_pred.shape[0],1,1,1)
            mask_target = self.mask_target.repeat(image_pred.shape[0],1,1,1)
            vtx_target = self.vtx_target.repeat(image_pred.shape[0],1,1)
            color_target = self.color_target.repeat(image_pred.shape[0],1,1)
        else:
            target = self.target
            mask_target = self.mask_target
            vtx_target = self.vtx_target
            color_target = self.color_target
        
        overlap_mask = mask_target * mask_pred
        union_mask = mask_target + mask_pred - overlap_mask
        self.iou = (overlap_mask.sum()/union_mask.sum()).item()
      
        loss = 0.0

        if self.args.mesh_regularization_loss:
            ### loss_flat in training
            vtx = self.mesh_template.get_vertex_positions(pred_mesh_map)
            flat_loss = loss_flat(self.mesh_template.mesh, self.mesh_template.compute_normals(vtx))
            loss += flat_loss * self.args.mesh_regularization_loss_wt

        if self.args.chamfer_mask_loss:
            ### no need to downsample here as vtx_pred is 482
            vtx_pred = others['vtx']
            d1, d2, idx1, idx2 = distChamfer(vtx_pred,vtx_target)
            cd = d1.mean(1) + d2.mean(1)
            loss += cd * self.args.chamfer_mask_loss_wt

        if self.args.chamfer_texture_pixel_loss:
            # NOTE: batch size should be one
            pix_pos_pred = mask2proj(mask_pred)
            pix_pred = grid_sample_from_vtx(pix_pos_pred, image_pred)
            dist_map_c, idx_a, idx_b = distChamfer_downsample(pix_pred,color_target,resolution=self.args.chamfer_resolution)
            dist_map_p, _, _ = distChamfer_downsample(pix_pos_pred,vtx_target,resolution=self.args.chamfer_resolution, idx_a=idx_a, idx_b=idx_b)

            xy_threshold = self.args.xy_threshold
            k = self.args.xy_k
            alpha = self.args.xy_alpha
            eps = 1 - (2*k*xy_threshold)**2
            rgb_eps = self.args.rgb_eps
            if eps == 1:
                xy_term = torch.pow(1+k*dist_map_p, alpha)
            else:
                xy_term = F.relu(torch.pow(eps+k*dist_map_p, alpha)-1) + 1
            dist_map = xy_term * (dist_map_c + rgb_eps)

            dist_min_ab = dist_map.min(-1)[0]
            dist_mean_ab = dist_min_ab.mean(-1)

            loss += dist_mean_ab * self.args.chamfer_texture_pixel_loss_wt
            
            ### colect the matched points in the target for visualization
            indices = dist_map.argmin(dim=-1)
            self.matched_pos = torch.stack([vtx_target[i,indices[i]] for i in range(indices.shape[0])],0)
            self.matched_clr = torch.stack([color_target[i,indices[i]] for i in range(indices.shape[0])],0)
            # v2 from: grid sample
            self.matched_clr_v2 = grid_sample_from_vtx(self.matched_pos, target) # NOTE that back vertices color shown as well

        if self.args.chamfer_texture_feat_loss:
            focal_feat_map_pred = self.vgg_feat(image_pred, mask_pred)
            focal_feat_map_target = self.vgg_feat(target, mask_target)
            subpool_dim = image_pred.shape[-1] // focal_feat_map_pred.shape[-1]
            this_mask_target_float = F.avg_pool2d(mask_target,subpool_dim)
            this_mask_pred_float = F.avg_pool2d(mask_pred,subpool_dim)
            this_mask_target = (this_mask_target_float >= self.args.subpool_threshold).type(this_mask_target_float.dtype)
            this_mask_pred = (this_mask_pred_float >= self.args.subpool_threshold).type(this_mask_pred_float.dtype)
            this_feat_pos_target = mask2proj(this_mask_target)
            this_feat_pos_pred = mask2proj(this_mask_pred)
            this_feat_target_unnorm = grid_sample_from_vtx(this_feat_pos_target, focal_feat_map_target)
            this_feat_pred_unnorm = grid_sample_from_vtx(this_feat_pos_pred, focal_feat_map_pred)
            this_feat_target = F.normalize(this_feat_target_unnorm, p=2, dim=2)
            this_feat_pred = F.normalize(this_feat_pred_unnorm, p=2, dim=2)

            chamfer_texture_feat_loss = self.compute_focal_chamfer(this_feat_pred,this_feat_pos_pred,this_feat_target,this_feat_pos_target)
            loss += chamfer_texture_feat_loss * self.args.chamfer_texture_feat_loss_wt
        
        if self.args.nll_loss:
            nll_loss = torch.mean(self.z**2 / 2, [1])
            loss += nll_loss * self.args.nll_loss_wt
            if self.curr_step >= 0:
                self.writer.add_scalar("z_norm", nll_loss, self.curr_step)

        return loss
    

    def init_z(self):
        """
        init for the z with smallest loss
        """
        self.curr_step = -1
        batch_size = min(self.args.init_batch_size, self.args.select_num)
        batches = int(self.args.select_num / batch_size)
        noise_ls = []
        loss_ls = []

        with torch.no_grad():
            for batch in range(batches):
                noise = torch.randn(batch_size, self.args.latent_dim).cuda()
                c, caption, others = self.prepare_input(batch_size=noise.shape[0])
                pred_tex, pred_mesh_map, attn_map = self.G(noise, c, caption, others, return_attention=True)

                image_pred, alpha_pred, others = self.render(pred_tex, pred_mesh_map, attn_map)
                loss = self.criterion(image_pred, alpha_pred, stage='init',others=others)
                noise_ls.append(noise)
                loss_ls.append(loss)

        losses = torch.cat(loss_ls,0)
        noises = torch.cat(noise_ls,0)

        idx = torch.argmin(losses)
        self.z.data = noises[idx].unsqueeze(0)

        return
        

    def run(self, ith=-1):

        self.curr_step = 0

        for stage, iteration in enumerate(self.args.iterations):

            for i in range(iteration):
                self.curr_step += 1
                # setup learning rate
                self.z_scheduler.update(self.curr_step, self.args.z_lrs[stage])
                if self.args.update_pose:
                    self.pose_scheduler.update(self.curr_step, self.args.pose_lrs[stage])

                # forward
                self.z_optim.zero_grad()
                if self.args.update_pose:
                    self.pose_optim.zero_grad()

                c, caption, others = self.prepare_input()
                pred_tex, pred_mesh_map, attn_map = self.G(self.z, c, caption, others, return_attention=True)

                image_pred, alpha_pred, others = self.render(pred_tex, pred_mesh_map, attn_map)
                
                loss = self.criterion(image_pred, alpha_pred, pred_tex=pred_tex, \
                    pred_mesh_map=pred_mesh_map, c=c, caption=caption,others=others)

                # backward
                loss.backward()
                self.z_optim.step()
                if self.args.update_pose:
                    self.pose_optim.step()
        

        ### save results, NOTE images with [-1,1] range
        if self.args.save_results:
            
            save_unit = {
                'z': self.z.detach().cpu(),
                'input_img': self.target.detach().cpu(),
                'input_mask': self.mask_target.detach().cpu(),
                'pred_img': image_pred.detach().cpu(),
                'pred_mask': alpha_pred.detach().cpu(),
                'pred_tex': pred_tex.detach().cpu(),
                'pred_shape': pred_mesh_map.detach().cpu(),
                'scale': self.scale.detach().cpu(),
                'translation':  self.translation.detach().cpu(),
                'rotation':  self.rotation.detach().cpu(),
                'img_path':  self.img_path,
                'idx':  self.idx,
            }

            results_dir = os.path.join('./outputs/inversion_results',self.args.name)
            if not os.path.exists(results_dir): 
                os.makedirs(results_dir)
            torch.save(save_unit, os.path.join(results_dir,str(self.idx)+'.pth'))
                    
        self.writer.flush()
        self.writer.close()


    def compute_focal_chamfer(self,feat_pred, feat_pred_pos, feat_target, feat_target_pos):
        dist_map_c, idx_a, idx_b = distChamfer_downsample(feat_pred,feat_target,resolution=self.args.chamfer_resolution)
        dist_map_p, _, _ = distChamfer_downsample(feat_pred_pos,feat_target_pos,resolution=self.args.chamfer_resolution, idx_a=idx_a, idx_b=idx_b)

        xy_threshold = self.args.xy_threshold
        k = self.args.xy_k
        alpha = self.args.xy_alpha
        eps = 1 - (2*k*xy_threshold)**2
        rgb_eps = self.args.rgb_eps
        if eps == 1:
            xy_term = torch.pow(1+k*dist_map_p, alpha)
        else:
            xy_term = F.relu(torch.pow(eps+k*dist_map_p, alpha)-1) + 1
        dist_map = xy_term * (dist_map_c + rgb_eps)
        dist_min_ab = dist_map.min(-1)[0]
        dist_mean_ab = dist_min_ab.mean(-1)

        dist_min_ba = dist_map.min(-2)[0]
        dist_mean_ba = dist_min_ba.mean(-1)
        this_loss = (dist_mean_ab + dist_mean_ba)/2
        return this_loss
    
