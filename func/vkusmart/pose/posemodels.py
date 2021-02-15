from typing import List, Tuple
import time
import sys

import numpy as np
import cv2
import torch

from vkusmart.types import Images, BoundingBoxes, BoundingBoxesWithScores, BB_KPs, Directory

from vkusmart.pose.ap_utils import im_to_torch, crop_from_dets, flip_v, shuffleLR, Mscoco, transformBoxInvert_batch
from vkusmart.pose.hourglass_duc_se import FastPose
from vkusmart.pose.pose_resnet import get_pose_net
from vkusmart.pose.pose_nms import pose_nms


class PoseModelMPPE(object):
    '''Multiple person pose estimation (MPPE) model interface
    '''
    def estimate(
        self, 
        imgs: Images, 
        bboxes_scores: List[BoundingBoxesWithScores], 
        extra_zoom: str=None,
        profile: bool=False,
        use_pose_nms: bool=True
    ) -> List[BB_KPs]:
        '''Estimates poses for given bounding boxes

        Args:
            imgs: list of images, len(imgs) == len(bboxes)
            bboxes_scores: bounding boxes and their scores from :py:funct:`.detectors.Detector.predict_with_scores`
            extra_zoom: type of camera from which images come from (works only for `OneStreamVideoProvider`). Options are: ['right_cam', 'left_cam']
        Returns:
            List of lists (BB, KP) shaped same as bboxes
        '''
        return self._estimate(
            imgs, 
            bboxes_scores, 
            extra_zoom=extra_zoom,
            profile=profile,
            use_pose_nms=use_pose_nms
        )

    
class AlphaPoseModel(PoseModelMPPE):
    '''MPPE pipeline borrowed from AlphaPose project (with Hourglass architecture).
       Includes all stages of multiple person keypoints prediction:
       1. Making crops from given bounding boxes
       2. Heatmaps from crops prediction
       3. Keypoints from heatmaps prediction
       4. Post-processing (pose_nms)
    '''
    def __init__(
        self,
        crop_width,
        crop_height,
        hm_model_name,
        hm_height,
        hm_width,
        weights_file,
        gpu_num,
        fast,
        dataset,
        config
    ):
        # desired persons' crop shape
        self.crop_height = crop_height
        self.crop_width = crop_width

        # SPPE heatmap model
        self.hm_model_name = hm_model_name
        self.hm_model_weights_path = weights_file
        # shape of the output heatmap tensor
        self.hm_height = hm_height
        self.hm_width = hm_width
        
        self.fast = fast
        self.gpu_num = gpu_num

        # load the heatmap model
        torch.cuda.set_device(self.gpu_num)
        print('Using GPU={}'.format(self.gpu_num))
        with torch.no_grad():
            if self.hm_model_name == 'hourglass':
                self.sppe_hm_model = FastPose().cuda(device='cuda:{}'.format(self.gpu_num))
            elif self.hm_model_name == 'pose_resnet50':
                self.sppe_hm_model = get_pose_net(cfg=config).cuda(device='cuda:{}'.format(self.gpu_num))
            else: 
                raise NotImplemented('Unknown model, options are: ["hourglass", "pose_resnet50"]')
            self.sppe_hm_model.load_state_dict(
                torch.load(
                    self.hm_model_weights_path, 
                    map_location=torch.device('cuda:{}'.format(self.gpu_num))
                )
            )
            self.sppe_hm_model.eval()
            print(f'Successfully loaded pretrained {self.hm_model_name} weights from {self.hm_model_weights_path}')
        
        # stuff for shuffleLR() inside of self._predict_heatmaps() to work
        self.dataset = Mscoco() if dataset == 'coco2017' else None
            
    def _predict_heatmaps(self, crops):
        '''Predict heatmaps with model borrowed from AlphaPose project (Hourglass architecture)
        
        Args:
            crops -- torch.Tensor of size `torch.Size([num_boxes, 3, self.crop_height, self.crop_width])`
        Returns:
            heatmaps -- torch.Tensor of size `torch.Size([num_boxes, 17, self.hm_height, self.hm_width])`
        '''
        if self.hm_model_name == 'pose_resnet_50':
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
            crops = normalize(crops)
            
        with torch.no_grad():
            heatmaps = self.sppe_hm_model(crops)
            heatmaps = heatmaps.narrow(1, 0, 17)

            if not self.fast:
                with torch.no_grad():
                    flipped_heatmaps = self.sppe_hm_model(flip_v(crops))
                flipped_heatmaps = flipped_heatmaps.narrow(1, 0, 17)
                flipped_heatmaps = flip_v(
                    shuffleLR(flipped_heatmaps, self.dataset)
                )
                heatmaps = (flipped_heatmaps + heatmaps) / 2

        return heatmaps
             
    def _predict_keypoints(self, heatmaps, bboxes, crop_height, crop_width, hm_height, hm_width):
        '''Predict keypoints for each heatmap
        
        Args:
            heatmaps -- torch.Tensor of size `torch.Size([num_boxes, 17, self.hm_height, self.hm_width])`
        
        '''
        assert heatmaps.dim() == 4, 'Score maps should be 4-dim'
        # main procedure
        maxval, idx = torch.max(heatmaps.view(heatmaps.size(0), heatmaps.size(1), -1), 2)

        maxval = maxval.view(heatmaps.size(0), heatmaps.size(1), 1)
        idx = idx.view(heatmaps.size(0), heatmaps.size(1), 1) + 1

        keypoints = idx.repeat(1, 1, 2).float()

        keypoints[:, :, 0] = (keypoints[:, :, 0] - 1) % heatmaps.size(3)
        keypoints[:, :, 1] = torch.floor((keypoints[:, :, 1] - 1) / heatmaps.size(3))

        pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
        keypoints *= pred_mask

        # Very simple post-processing step to improve performance at tight PCK thresholds
        for i in range(keypoints.size(0)):
            for j in range(keypoints.size(1)):
                hm = heatmaps[i][j]
                pX, pY = int(round(float(keypoints[i][j][0]))), int(round(float(keypoints[i][j][1])))
                if 0 < pX < self.hm_width - 1 and 0 < pY < self.hm_height - 1:
                    diff = torch.Tensor(
                        (hm[pY][pX + 1] - hm[pY][pX - 1], hm[pY + 1][pX] - hm[pY - 1][pX])
                    ).cuda(device='cuda:{}'.format(self.gpu_num))
                    keypoints[i][j] += diff.sign() * 0.25
        keypoints += 0.2

        # I don't know a purpose of this function :(
        keypoints_transformed = torch.zeros(keypoints.size())
#         print('bboxes shape:', bboxes.shape)
        keypoints_transformed = transformBoxInvert_batch(
            keypoints, 
            bboxes[:,:2], 
            bboxes[:,2:], 
            crop_height, 
            crop_width, 
            hm_height, 
            hm_width
        )

        return keypoints, keypoints_transformed, maxval
            
    def _pose_nms(self, *args, **kwargs):
        return pose_nms(*args, **kwargs)
        
    def _estimate(
        self, 
        imgs: Images, 
        all_bboxes_scores: List[BoundingBoxesWithScores], 
        extra_zoom: str, 
        profile: bool,
        use_pose_nms: bool
    ) -> List[BB_KPs]:
        '''Estimates poses for given bounding boxes

        Args:
            imgs: list of images, len(imgs) == len(bboxes)
            all_bboxes_scores: bounding boxes and their scores from :py:funct:`.detectors.Detector.predict_with_scores`
            extra_zoom: wheter to expand bboxes for arms to lie inside of them
            profile: wheter to print info about time taken by each component
        Returns:
            bb_kps - list of lists (BB, KP) with same length as all_bboxes_scores
        '''
        
        # keypoints for crops from all imgs
        all_keypoints = []
        
        for img, bboxes_scores in zip(imgs, all_bboxes_scores):
            if len(bboxes_scores) == 0:
                all_keypoints.append({})
                continue
            
            img_torch_rgb = im_to_torch(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # return crops (parts of images inside bboxes) and extended (zoomed) bboxe
            # ADD EXTRA EXPANSION BECAUSE OF ARMS OUT OF THE BOX
            only_bboxes = [bbox_score[0] for bbox_score in bboxes_scores]
            if profile: begin = time.time()
            crops_zoomed, bboxes_zoomed = crop_from_dets(
                img_torch_rgb, 
                only_bboxes, 
                self.crop_height, 
                self.crop_width,
                extra_zoom=extra_zoom
            )
            if profile: print('crop time:', time.time() - begin)
                
            crops_zoomed = crops_zoomed.cuda(device='cuda:{}'.format(self.gpu_num))
            bboxes_zoomed = bboxes_zoomed.cuda(device='cuda:{}'.format(self.gpu_num))
#             print('crops_zoomed:', crops_zoomed.shape)
#             print('bboxes_zoomed:', bboxes_zoomed.shape)
            
            # predict heatmaps for given zoomed crops
            if profile: begin = time.time()
            crops_heatmaps = self._predict_heatmaps(crops_zoomed)
            if profile: print('heatmaps time:', time.time() - begin)
            
            # location prediction (n, kp, 2) | score prediction (n, kp, 1)
            # predict keypoints and their scores using heatmaps from FastPose
            if profile: begin = time.time()
            preds_hm, preds_img, preds_scores = self._predict_keypoints(
                crops_heatmaps, 
                bboxes_zoomed,
                self.crop_height,
                self.crop_width,
                self.hm_height,
                self.hm_width
            )
            if profile: print('keypoints time:', time.time() - begin)
#             print(preds_hm.shape, preds_img.shape, preds_scores.shape)
            del crops_heatmaps  # !!!
            torch.cuda.empty_cache()  # !!!

            only_scores = torch.from_numpy(np.array([bbox_score[1] for bbox_score in bboxes_scores]))
            if profile: begin = time.time()
#             keypoints is a dict with:
#             {
#                 'keypoints': tensor.Size(num_boxes, 17, 2), 
#                 'kp_score': tensor.Size(num_boxes, 17, 1), 
#                 'proposal_score': tensor.Size(1)
#             }
            if use_pose_nms:
                keypoints = self._pose_nms(
                    bboxes_zoomed.cpu().detach(), 
                    only_scores.cpu().detach(), 
                    preds_img.cpu().detach(), 
                    preds_scores.cpu().detach()
                )
            else:
                keypoints = []
                for kps, scores in zip(preds_img.cpu().detach(), preds_scores.cpu().detach()):
                    keypoints_record = {}
                    keypoints_record['keypoints'] = kps
                    keypoints_record['kp_score'] = scores
                    keypoints_record['proposal_score'] = torch.IntTensor([-1])
                    keypoints.append(keypoints_record)
            if profile: print('pose_nms time:', time.time() - begin, '\n')

            all_keypoints.append(keypoints)
        
        bb_kps = [
            [
                [
                    bbox_score[0] if len(bbox_score) else [], 
                    person_keypoints if len(keypoints) else {}
                ] for bbox_score, person_keypoints in zip(bboxes_scores, image_keypoints)
            ] for bboxes_scores, image_keypoints in zip(all_bboxes_scores, all_keypoints)
        ]
        
        return bb_kps
