

import torch
import torch.nn as nn
import torchvision
from . import cmr_net_blocks as nb
#------------- Modules ------------#
#----------------------------------#
class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        return x

class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, n_blocks=4, nz_feat=100, batch_norm=True):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(n_blocks=4)
        self.enc_conv1 = nb.conv2d(batch_norm, 512, 256, stride=2, kernel_size=4)
        nc_input = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)
        self.enc_fc = nb.fc_stack(nc_input, nz_feat, 2)

        nb.net_init(self.enc_conv1)

    def forward(self, img):
        resnet_feat = self.resnet_conv.forward(img)

        out_enc_conv1 = self.enc_conv1(resnet_feat)
        out_enc_conv1 = out_enc_conv1.view(img.size(0), -1)
        feat = self.enc_fc.forward(out_enc_conv1)

        return feat

class QuatPredictor(nn.Module):
    def __init__(self, nz_feat, nz_rot=4, classify_rot=False):
        super(QuatPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, nz_rot)
        self.classify_rot = classify_rot

    def forward(self, feat):
        quat = self.pred_layer.forward(feat)
        if self.classify_rot:
            quat = torch.nn.functional.log_softmax(quat)
        else:
            quat = torch.nn.functional.normalize(quat)
        return quat


class ScalePredictor(nn.Module):
    def __init__(self, nz):
        super(ScalePredictor, self).__init__()
        self.pred_layer = nn.Linear(nz, 1)

    def forward(self, feat):
        scale = self.pred_layer.forward(feat) + 1  #biasing the scale to 1
        scale = torch.nn.functional.relu(scale) + 1e-12
        return scale


class TransPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """

    def __init__(self, nz, orth=True):
        super(TransPredictor, self).__init__()
        if orth:
            self.pred_layer = nn.Linear(nz, 2)
        else:
            self.pred_layer = nn.Linear(nz, 3)

    def forward(self, feat):
        trans = self.pred_layer.forward(feat)
        return trans


class CamPoseEstimator(nn.Module):
    def __init__(self, nz_feat=200, input_shape=(256,256), args=None):
        
        super(CamPoseEstimator, self).__init__()
        # self.model = mesh_net.MeshNet(
        #     img_size, args, nz_feat=args.nz_feat, num_kps=args.num_kps, sfm_mean_shape=sfm_mean_shape)
        # class MeshNet(nn.Module):
             # def __init__(self, input_shape, args, nz_feat=100, num_kps=15, sfm_mean_shape=None):
        # self.code_predictor = CodePredictor(nz_feat=nz_feat, num_verts=self.num_output,args=args)
        self.encoder = Encoder(input_shape=input_shape, n_blocks=4, nz_feat=nz_feat)
        self.quat_predictor = QuatPredictor(nz_feat)
        
        self.scale_predictor = ScalePredictor(nz_feat)
        self.trans_predictor = TransPredictor(nz_feat)

    def forward(self,x):
        feat = self.encoder.forward(x)
        scale_pred = self.scale_predictor.forward(feat)
        quat_pred = self.quat_predictor.forward(feat)
        trans_pred = self.trans_predictor.forward(feat)

        return scale_pred, trans_pred, trans_pred
