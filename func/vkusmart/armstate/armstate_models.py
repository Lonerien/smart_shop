from collections import defaultdict
from pathlib import Path
import numpy as np
import PIL
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms

from efficientnet_pytorch import EfficientNet

from vkusmart.armstate.armstate_utils import *

DEBUG_ARMSTATE_MODELS = False

class ArmstateClassifier:
    '''  
    Class for writing prediction for arm to bb_ids_kps
    
    Parameters: 
    arch: one of torchvision classification models (resnet18, resnet50, ...)
    gpu_num: gpu_num to which to move tensors (cuda number or cpu)

    Returns: see method `classify()` for details
    
    Usage: /mnt/nfs/vkusmart/master/ks_picker_count.ipynb
    '''
    def __init__(
        self, 
        arch, 
        num_classes,
        crop_size,
        gpu_num, 
        weights_path,
    ):
        self.arch = arch
        self.num_classes = num_classes
        self.crop_size = crop_size
        self.gpu_num = gpu_num
        self.weights_path = weights_path

        with torch.no_grad():
            if self.arch == 'resnet18':
                # old version from Kseniya Valchuk
                #self.model = models.resnet18(num_classes=self.num_classes)
                # new version from Arkady Ilin
                self.model = models.resnet18(pretrained=True)
                num_features = self.model.fc.in_features
                self.model.fc = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(256, 4)
                )
            elif self.arch == 'resnet50':
                self.model = models.resnet50(pretrained=True)
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, self.num_classes)
            elif self.arch == 'efficientnet-b0':
                # Implementation: https://github.com/lukemelas/EfficientNet-PyTorch
                self.model = EfficientNet.from_pretrained(self.arch)
                num_features = self.model._fc.in_features
                self.model._fc = nn.Sequential(
                    nn.Linear(num_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(256, 4)
                ) 
            elif self.arch == 'mobilenet-v2':
                self.model = models.mobilenet_v2(pretrained=True)
                self.model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(
                        in_features=1280, 
                        out_features=self.num_classes, 
                        bias=True
                    )
                )
            elif self.arch == 'mobilenet-v2-compl':
                self.model = models.mobilenet_v2(pretrained=True)
                self.model.classifier = nn.Sequential(
                        nn.Linear(1280, 256),
                        nn.ReLU(),
                        nn.Dropout(p=0.4),
                        nn.Linear(
                            in_features=256, 
                            out_features=self.num_classes, 
                            bias=True)
                    )
            else:
                info_text = 'Unknown model, options are: ["resnet18", "resnet50", "efficientnet-b0", "mobilenet-v2", "mobilenet-v2-compl"]'
                raise NotImplemented(info_text)
                
            checkpoint = torch.load(
                self.weights_path, 
                map_location=torch.device('cuda:{}'.format(self.gpu_num))
            )
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict)
            self.model = self.model.to('cuda:{}'.format(gpu_num))
            self.model.eval()
            
            print(f'Using GPU={self.gpu_num}')
            print(f'Successfully loaded pretrained {self.arch} weights from {self.weights_path}')   

    def _process_image(self, image):
        if self.arch == 'resnet18' and self.num_classes == 3:
            # old setting (deprecated)
            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((125,125)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
            ])
        image = preprocess(image)
        return image

    def _predict(self, img):
        ''' Predict probability and class of image
        '''
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)
        img = img.to('cuda:{}'.format(self.gpu_num))
        logits = self.model(img)
        ps = F.softmax(logits, dim=1)
        probs = ps.cpu().data.numpy().squeeze()
        return (probs, probs.argmax())

    def _make_crops(self, person, frame, crop_size):
        '''Makes crops of person's arms.

        Parameters:
        person: one detection of person from bb_ids_kps[cam_id]
        frame: frame (image) from one camera

        Returns:
        record: dict with fields: track_id, bbox, keypoints_info and crops_info
        '''
        record = defaultdict(dict)
        keypoint_indices = {'left': (7, 9), 'right': (8,10)}
        for arm_type in ['left', 'right']:
            elbow_idx, wrist_idx = keypoint_indices[arm_type]
#             print(person)
            elbow_coords = [int(x) for x in person['keypoints_info']['keypoints'][elbow_idx]]
            wrist_coords = [int(c) for c in person['keypoints_info']['keypoints'][wrist_idx]]
            # re-calculate the coords for better accuracy and get the crop coords
            crop_coords = f(
                elbow_coords, wrist_coords, 
                normal_arm_length=100, 
                min_size=crop_size, 
                max_size=crop_size
            )
            crop_img = crop(frame, *crop_coords)
            record['crops_info'][f'{arm_type}_crop'] = crop_img
            # xywh -> tlbr
            record['crops_info'][f'{arm_type}_coords'] = (
                crop_coords[0], 
                crop_coords[1], 
                crop_coords[0]+crop_coords[2],
                crop_coords[1]+crop_coords[3]
            )
        record['track_id'] = person['track_id']
        record['bbox'] = person['bbox']
        record['keypoints_info'] = person['keypoints_info']
        return record

    # frames from shelf_provider (frames from several cameras)
    def classify(self, bb_ids_kps, frames):
        ''' Writes predictions into bb_ids_kps

        Parameters:
        bb_ids_kps: 
        frames: images from shelf_provider

        Returns:
        frame_crop_track_status_kps: extended bb_ids_kps with arm crops information
        '''
        frame_crop_track_status_kps = []
        for cam_id, frame in enumerate(frames):
            camera_crop_track_status_kps = []
            for person in bb_ids_kps[cam_id]:
                crop_track_kps = self._make_crops(person, frame, crop_size=self.crop_size)
                crop_track_status_kps = crop_track_kps
                for arm_type in ['left', 'right']:
                    if crop_track_kps['crops_info'][f'{arm_type}_crop'] is not None:
                        try:
                            crop = self._process_image(crop_track_kps['crops_info'][f'{arm_type}_crop'])
                            full_probs, label = self._predict(crop)
#                             print(crop_name, full_probs, label, prob)
                        except ValueError:
                            print('armstate_model.py: Arm crop is out of image!')
                            print('Assigning the "2" value (trash class) to this arm`s state')
                            full_probs, label = np.array([0,0,1.0,0]), 2
                           
                        crop_track_status_kps['crops_info'][f'{arm_type}_state'] = label 
                        crop_track_status_kps['crops_info'][f'{arm_type}_class_probs'] = tuple(np.round(full_probs, 4))
                        
                        if DEBUG_ARMSTATE_MODELS:
                            print('If scenario, crop_track_status_kps: ', 
                                  crop_track_status_kps['crops_info'][f'{arm_type}_class_probs'])
                    else:    
                        crop_track_status_kps['crops_info'][f'{arm_type}_state'] = None
                        crop_track_status_kps['crops_info'][f'{arm_type}_class_probs'] = None
                        
                        if DEBUG_ARMSTATE_MODELS:
                            print('Else scenario, crop_track_status_kps: ', 
                                  crop_track_status_kps['crops_info'][f'{arm_type}_class_probs'])
                        
                camera_crop_track_status_kps.append(crop_track_status_kps)
            frame_crop_track_status_kps.append(camera_crop_track_status_kps)
        return frame_crop_track_status_kps
