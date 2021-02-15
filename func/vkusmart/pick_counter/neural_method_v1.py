import numpy as np
import torch
from torch import nn

class PickNetClassifier:
    def __init__(self, path_to_weights: str, num_features: int, device:int ):
        ''' Args:
        path_to_model -- path to weights of classification model 
        num_features -- number of features
        device -- torch device
        '''
        # test conf: 
        # path_to_weights: /mnt/nfs/user6/model.pt
        # num_features: 7
        # device: 1
        
        self.path = path_to_weights
        self.num_features = num_features
        self.model = ResNetTriple(BasicBlock, [2, 2, 2, 2], num_features=num_features)
        self.device = 'cuda:'+str(device) # cuda:1
        
        self.model.load_state_dict(torch.load(self.path)) 
        self.model.to(self.device)
        print('NN was Inited!')

    def predict(self, x):
        ''' Input:
        x: numpy array of features; shape = (1, seq_len, num_features), where
            seq_len -- length of frame sequence (default == 25)
            num_features -- number of features (default == 11):
                'vx' -- arm speed for x-axis,
                2'vy' -- arm speed for y-axis,
                1'v' -- vx^2 + vy^2, 
                0'module_to_shelf' -- arm speed relative to the line of the shelf, 
                'distance_to_shelf' -- distance to shelf line, 
                'distance_to_hip' -- distance to hip,
                3,4,5,6'arm_state' -- arm class predicted by classifier,
                'class_empty_prob' -- arm probability to be empty,
                'class_product_prob' -- arm probability to be with a product,
                'class_trash_prob' -- probability of trash (no arm on the crop, not visible and so on),
                'class_bag_prob' -- arm probability to hold a bag
                
        Output: probabilities of 'take' and 'left' actions for the MIDDLE frame of the sequence
        '''
        print('================PREDICT BEGIN================')
#         x = x.transpose((0,2,1))  # features == channels
        inputs = torch.from_numpy(x[:,:,:]).float().to(self.device)
        print('inputs.shape = ', inputs.shape)
        
#         self.model.load_state_dict(torch.load(self.path)) # мб вынести в init
        self.model.eval()
        
        class_score = self.model(inputs)

        predicted_take = class_score[:, 1].item()
        predicted_left = class_score[:, 2].item()
        print('================PREDICT END================')
        print('predicted_take, predicted_left', predicted_take, predicted_left)
        print('===RESULT===')
        
        return predicted_take, predicted_left # [0,1] вероятность взятия, [0,1] вероятность возврата
    


def conv3x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    3x3 convolution with padding
    padding == dilation in order to get the same input size
    """
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x1(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetTriple(nn.Module):
    '''Modified (1D) version of 
     https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
     '''
    def __init__(self, block, layers, num_features, num_classes=3, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetTriple, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            """
            Maybe we should try dilated convolution
            """
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(num_features, self.inplanes, kernel_size=7, 
                               stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
#         print(x.shape)
        x = self.conv1(x)
#         print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
#         print(x.shape)
        
        x = self.layer1(x)
#         print(x.shape)
        x = self.layer2(x)
#         print(x.shape)
        x = self.layer3(x)
#         print(x.shape)
        x = self.layer4(x)
#         print(x.shape)
        
        x = self.avgpool(x)
#         print(x.shape)
        x = torch.flatten(x, 1)
#         print(x.shape)
        x = self.fc(x)
        
        pick_proba = self.softmax(x)
        
        return pick_proba