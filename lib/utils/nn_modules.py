

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std, input_range='01'):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        # self.mean = torch.tensor(mean).view(-1, 1, 1)
        # self.std = torch.tensor(std).view(-1, 1, 1)
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)
        self.input_range = input_range

    def forward(self, img):
        if self.input_range == 'n11':
            img = img/2. + 0.5
        # normalize img
        return (img - self.mean) / self.std

class FeaturesRes18(nn.Module):
    """
    backbone: resnet18
    """

    def __init__(self):
        super(FeaturesRes18, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        # resnet18.load_state_dict(torch.load('resnet18.pth'))
        modules = list(resnet18.children())[:-1]
        self.features = nn.Sequential(*modules)

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        return output