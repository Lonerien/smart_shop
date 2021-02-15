import visdom
import time
import sys

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed

from vkusmart.pose.ap_utils import flip_v, shuffleLR
from vkusmart.pose.se_resnet import SEResnet

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    

class DUC(nn.Module):
    '''
    INPUT: inplanes, planes, upscale_factor
    OUTPUT: (planes // 4) * ht * wd
    '''
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class FastPose(nn.Module):
    '''
    # parser.add_argument('--nFeats', default=256, type=int,
    #                     help='Number of features in the hourglass')
    # parser.add_argument('--nClasses', default=33, type=int,
    #                     help='Number of output channel')
    # parser.add_argument('--nStack', default=4, type=int,
    #                     help='Number of hourglasses to stack')
    '''
    DIM = 128

    def __init__(self, num_classes=33):
        super(FastPose, self).__init__()

        # backbone network
        self.preact = SEResnet('resnet101')

        self.shuffle = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2)
        self.duc2 = DUC(256, 512, upscale_factor=2)

        self.conv_out = nn.Conv2d(
            self.DIM, num_classes, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Variable):
        out = self.preact(x)
        out = self.shuffle(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        return out

    
# class InferNet(nn.Module):
#     def __init__(self, kernel_size, dataset, weights_path, gpu_num, fast=False):
#         super(InferNet, self).__init__()

#         torch.cuda.set_device(gpu_num)
        
#         self.hm_model = FastPose().cuda()
#         print('Loading SPPE heatmap prediction model from {}'.format(weights_path))
        
#         self.hm_model.load_state_dict(
#             torch.load(
#                 weights_path, 
#                 map_location=torch.device('cuda:{}'.format(gpu_num)
#             )
#         )
#         self.hm_model.eval()
            
#         self.dataset = dataset
#         self.fast = fast

#     def forward(self, x):
#         out = self.hm_model(x)
#         out = out.narrow(1, 0, 17)

#         if not self.fast:
#             flip_out = self.hm_model(flip_v(x))
#             flip_out = flip_out.narrow(1, 0, 17)

#             flip_out = flip_v(
#                 shuffleLR(flip_out, self.dataset)
#             )

#             out = (flip_out + out) / 2

#         return out
