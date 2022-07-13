from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import scipy.misc
import tqdm
import cv2

import torch

import rendering.cmr_geom_utils as geom_utils

from kaolin.graphics.NeuralMeshRenderer import NeuralMeshRenderer

class NeuralRenderer(torch.nn.Module):
    """
    replace NeuralRenderer from nmr.py with the kaolin's
    """
    def __init__(self, img_size=256,uv_sampler=None):
        super(NeuralRenderer, self).__init__()
        self.renderer = NeuralMeshRenderer(image_size=img_size, camera_mode='look_at',perspective=False,viewing_angle=30,light_intensity_ambient=0.8)
        # 30 degree is equivalent to self.renderer.eye = [0, 0, -2.732]

        self.offset_z = 5.
        self.proj_fn = geom_utils.orthographic_proj_withz
        print('NMR-kaolin initiated')

    def ambient_light_only(self):
        # Make light only ambient.
        self.renderer.light_intensity_ambient = 1
        self.renderer.light_intensity_directional = 0
    
    def set_bgcolor(self, color):
        self.renderer.background_color = color

    def project_points(self, verts, cams):
        proj = self.proj_fn(verts, cams)
        return proj[:, :, :2]
    
    def forward(self, vertices, faces, cams, textures=None):
        # faces: B, 1280, 3
        # vertices:[B, 642, 3]
        # texture: texture: [B, 1280, 6, 6, 6, 3] - should be RGB, with range [0,1]
        # if textures is not None:
            
        verts = self.proj_fn(vertices, cams, offset_z=self.offset_z)
        
        vs = verts.clone()
        vs[:, :, 1] *= -1
        fs = faces.clone()
        if textures is None:
            self.mask_only = True
            masks = self.renderer.render_silhouettes(vs,fs)
            return masks
        else:
            self.mask_only = False
            ts = textures.clone()
            # print('tx shape:',ts.shape)
            imgs = self.renderer.render(vs, fs, ts)[0] #only keep rgb, no alpha and depth
            # for i, img in enumerate(imgs):

            #     img = img.permute([1,2,0]).detach().cpu().numpy()
            #     
            #     cv2.imwrite('./vis/nmr'+str(i)+'.jpg',img*255)
            #     print('saved img')
            # print('!!!imgs:',imgs.shape)
            return imgs