from typing import List, Tuple
import time
import sys

import numpy as np
import cv2
import torch

from vkusmart.types import Images, BoundingBoxes, BB_FVs, Directory

# This is OCHEN BIG KOSTIL, be careful, rebyatushki...
open_reid_path = './open-reid/reid/'
if open_reid_path not in sys.path:
    sys.path.append(open_reid_path)
#import reid  # OpenReid

deep_reid_path = '/mnt/nfs/vkusmart/reid/deep-person-reid/'
if deep_reid_path not in sys.path:
    sys.path.append(deep_reid_path)
import torchreid  # DeepPersonReid

class Reid(object):
    '''Person reidentification interface
    '''
    def extract(self, imgs: Images, bboxes: List[BoundingBoxes]) -> List[BB_FVs]:
        '''Computes embeddings for given bounding boxes

        Args:
            imgs: list of images, len(imgs) == len(bboxes)
            bboxes: bounding boxes from :py:funct:`.detectors.Detector.predict`
        Returns:
            List of lists (BB, FV) shaped same as bboxes
        '''
        return self._extract(imgs, bboxes)


class TorchReid(Reid):
    '''Person reidentification based on torchreid repo
    '''
    RESUME_PATHS = {
        'resnet50mid': {
            'softmax': {
                'vkusmart_v2_am': '/mnt/nfs/vkusmart/reid/deep-person-reid/log/resnet50mid-softmax-vkusmart_v2-staged_lr_adam003/model.pth.tar-60'
            }
        }
    }
    NUM_CLASSES = {
        'vkusmart_v4_am': 363
    }
    
    def __init__(
        self, 
        arch: str,
        loss: str,
        dataset: str,
        gpu_num: int,
        #feature_layer: str,
        weights_path: str,
        verbose=False
    ):
        self.arch = 'osnet_ibn_x1_0'
        self.loss = loss
        self.dataset = dataset
        self.gpu_num = gpu_num
        self.verbose = verbose
        self.weights_path = weights_path
        self.width = 128
        self.height = 256
        
        torch.cuda.set_device(self.gpu_num)
        with torch.no_grad():
            print('Using GPU={}'.format(self.gpu_num))
            print('torchreid.__version__ = ', torchreid.__version__)
            self.model = torchreid.models.build_model(
                name=self.arch,
                num_classes=self.NUM_CLASSES[self.dataset],
                loss=self.loss,
                pretrained=True
    #             use_stn=False,  # Spatial Transformer Network
    #             stn_only=False
            )   
            if self.verbose:
                print('DeepReid {} model trained on {} is created.\n'.format(self.arch, self.dataset))
                print(self.model)
#             weight_path = self.RESUME_PATHS[self.arch][self.loss][self.dataset]
            torchreid.utils.load_pretrained_weights(
                self.model, 
                self.weights_path
            )
            self.model.to('cuda:{}'.format(self.gpu_num))
            self.model.eval()

        if self.verbose:
            print("Weights are successfully loaded.")
        
    def print_model_summary(self):
        print(self.model)
                  
    def _extract(self, imgs, bboxes, verbose=False):
        '''see :py:funct:`.Reid.extract`'''
        result = []

        for img, boxes in zip(imgs, bboxes):
            self.output = []
            crops = []
            
            if not len(boxes):  # no boxes on the image
                result.append([])
                continue
            
            if verbose:
                print('_extract() len(boxes):', len(boxes))
            for box in boxes:
                if verbose:
                    print('Box', box, 'img.shape',  img.shape)
                crop = img[box[1]:box[3], box[0]:box[2], :]
                resized_crop = cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_AREA)
                crops.append(resized_crop)
            
            numpy_crops = np.array(crops)
            tensor_crops = torch.FloatTensor(numpy_crops.transpose(0, 3, 1, 2))  # channels first for PyTorch Conv2d()

            begin = time.time()
            if self.verbose:
                print(tensor_crops.shape)
            tensor_crops = tensor_crops.to('cuda:{}'.format(self.gpu_num))
            features = self.model(tensor_crops)
            if self.verbose:
                print(features.shape)
            if self.verbose:
                print('Reid time:', time.time() - begin)

            numpy_features = features.cpu().detach().numpy()
            result.append(list(zip(boxes, numpy_features)))

        return result

    
class OpenReid(Reid):
    '''Person reidentification based on open-reid repo
    '''
    SHAPES = {
        'resnet50': (256, 128)
    }
    RESUME_DIRS = {
        'resnet50': {
            'softmax': {
                'msmt17': 'vkusmart/open-reid/logs/softmax-loss/msmt17-resnet50/model_best.pth.tar',
                'vkusmart_v1': 'vkusmart/open-reid/logs/softmax-loss/vkusmart_v1-resnet50/model_best.pth.tar',
                'vkusmart_v2': 'vkusmart/open-reid/logs/softmax-loss/vkusmart_v2-resnet50/model_best.pth.tar',
                'vkusmart_v2_augmented': 'vkusmart/open-reid/logs/softmax-loss/vkusmart_v2_augmented-resnet50/model_best.pth.tar',
                'vkusmart_v2_augmented_mean': 'vkusmart/open-reid/logs/softmax-loss/vkusmart_v2_augmented_mean/model_best.pth.tar'
            },
            'triplet': {
                'vkusmart_v2_augmented_mean': 'vkusmart/open-reid/logs/triplet-loss/vkusmart_v2_am_feats128_wdecay0.0005_drop0_epochs150/model_best.pth.tar'
            }
        }
    }
    NUM_CLASSES = {
        'msmt17': 941,
        'vkusmart_v1': 72,
        'vkusmart_v2': 167,
        'vkusmart_v2_augmented': 167,
        'vkusmart_v2_augmented_mean': 167
    }
    ODD_LAYERS = {
        'resnet50': {
            'special_linear': [
                'feat_bn.weight', 
                'feat_bn.bias', 
                'feat_bn.running_mean',
                'feat_bn.running_var',
                'feat_bn.num_batches_tracked',
                'classifier.weight', 
                'classifier.bias'
            ],
            'special_linear_bn': [
                
                'classifier.weight', 
                'classifier.bias'
            ],
            'classifier': [
            ]
        }
    }
    
    def __init__(
        self,
        arch: str='resnet50',
        dataset: str='vkusmart_v2_augmented_mean',
        num_features: int=128,
        dropout_proba: float=0.5,
        cut_at_pooling: bool=False,
        gpu_num: int=0,
        feature_layer='classifier',
        loss='softmax',
        *,
        verbose=False,
    ):
        '''
        Args:
            arch: reid-model architecture, available options are: 'resnet50'
            dataset: name of the dataset to take trained weights from, 
             available optins are: 'msmt17', 'vkusmart_v1'
            num_features: int, dimension of feature vector for each crop
            dropout_proba: only for training, see open-reid docs
            cut_at_pooling: whether cut the model at GlobalPooling
            gpu_num: index of cuda device to load the model on
            verbose: to print current status to standard output or not
        '''
        self.arch = arch
        self.dataset = dataset
        self.num_features = num_features
        self.dropout_proba = dropout_proba
        # only feature extraction
        self.num_classes = 0 if not feature_layer == 'classifier' else OpenReid.NUM_CLASSES[self.dataset]
        self.cut_at_pooling = cut_at_pooling
        self.gpu_num = gpu_num
        self.feature_layer = feature_layer
        self.loss = loss
        self.verbose = verbose
        
        self.resume_dir = OpenReid.RESUME_DIRS[self.arch][self.loss][self.dataset]

        self.height, self.width = OpenReid.SHAPES[self.arch]

        # create the model architecture
        if self.loss == 'softmax':
            self.model = reid.models.create(
                self.arch,
                num_features=self.num_features,
                dropout=self.dropout_proba, 
                num_classes=self.num_classes,
                cut_at_pooling=self.cut_at_pooling
            )
        elif self.loss == 'triplet':
            self.model = reid.models.create(
                self.arch, 
                num_features=1024,
                dropout=self.dropout_proba,
                num_classes=self.num_features,
                cut_at_pooling=self.cut_at_pooling
            )
        self.model.eval()
        if self.verbose:
            print('OpenReid {} model trained on {} is created.'.format(self.arch, self.dataset))
            
        print(self.model)

        # load the best model weights
        checkpoint = reid.utils.serialization.load_checkpoint(
            fpath=self.resume_dir, 
            gpu_num=self.gpu_num
        )
        
#         print(checkpoint['state_dict'].keys())
        for l in OpenReid.ODD_LAYERS[self.arch][self.feature_layer]:
            checkpoint['state_dict'].pop(l)
#         for name in checkpoint['state_dict']:
#             print(name, checkpoint['state_dict'][name])
        
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to('cuda:{}'.format(self.gpu_num))
        self.model.eval()
        
        start_epoch = best_top1 = 0
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        if self.verbose:
            print("Weights are loaded.\nLast epoch: {}\nbest top1 {:.1%} (strange number, shouldn't be considered)"
              .format(start_epoch, best_top1))

    def print_model_summary(self):
        print(self.model)

    def _extract(self, imgs, bboxes, verbose=False):
        '''see :py:funct:`.Reid.extract`'''
        result = []

        for img, boxes in zip(imgs, bboxes):
            self.output = []
            crops = []
            
            if not len(boxes):  # no boxes on the image
                result.append([])
                continue
            
            if verbose:
                print('_extract() len(boxes):', len(boxes))
            for box in boxes:
                if verbose:
                    print('Box', box, 'img.shape',  img.shape)
                crop = img[box[1]:box[3], box[0]:box[2], :]
                resized_crop = cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_AREA)
                crops.append(resized_crop)
            
            numpy_crops = np.array(crops)
            tensor_crops = torch.FloatTensor(numpy_crops.transpose(0, 3, 1, 2))  # channels first for PyTorch Conv2d()

            begin = time.time()
            if self.verbose:
                print(tensor_crops.shape)
            tensor_crops = tensor_crops.to('cuda:{}'.format(self.gpu_num))
            features = self.model(tensor_crops)
            if self.verbose:
                print(features.shape)
            if self.verbose:
                print('Reid time:', time.time() - begin)

            numpy_features = features.cpu().detach().numpy()
            result.append(list(zip(boxes, numpy_features)))

        return result
