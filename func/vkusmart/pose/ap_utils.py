import numpy as np
import scipy.misc
import torch
import cv2

from torchsample.transforms import SpecialCrop, Pad
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.data as data

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def cropBox(img, up_left, bottom_right, target_height, target_width):
    up_left = up_left.int()
    bottom_right = bottom_right.int()
    
    curr_height = max(
        bottom_right[1] - up_left[1], 
        (bottom_right[0] - up_left[0]) * target_height / target_width
    )
    curr_width = curr_height * target_width / target_height
    if img.dim() == 2:
        img = img[np.newaxis, :]

    new_dim = torch.IntTensor((img.size(0), int(curr_height), int(curr_width)))
    new_img = img[:, up_left[1]:, up_left[0]:]
    
    # crop and padding
    size = torch.IntTensor((bottom_right[1] - up_left[1], bottom_right[0] - up_left[0]))
    new_img = SpecialCrop(size, 1)(new_img)
    new_img = Pad(new_dim)(new_img)
    
    # resize to output
    v_img = torch.unsqueeze(new_img, 0)
    # newImg = F.upsample_bilinear(v_Img, size=(int(resH), int(resW))).data[0]
    new_img = F.upsample(
        v_img, 
        size=(int(target_height), int(target_width)),
        mode='bilinear', 
        align_corners=True
    ).data[0]
    
    return new_img


def crop_from_dets(
    img, 
    bboxes, 
    target_height, 
    target_width,
    extra_zoom
):
    """
    Crop human from origin image according to Dectecion Results
    
    :args:
        img -- input image of size `torch.Size([3, H, W])` (channels first)
        bboxes -- List[Tuple[BoundingBox]] for this img
        target_height -- height which will have the final crops
        target_width -- width which will have the final crops
    :returns:
        extended (zoomed) boxes with scores and crops made from them
    """

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    # normalization (per-channel)
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    
    crops = []
    bboxes_zoomed = []
    for box in bboxes:
        upLeft = torch.Tensor(
            (float(box[0]), float(box[1])))
        bottomRight = torch.Tensor(
            (float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]
        if width > 100:
            scaleRate = 0.2
        else:
            scaleRate = 0.3

        # zooming the predicted bounding box
        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)
        
        # ADD EXTRA EXPANSION BECAUSE OF ARMS OUT OF THE BOX !!!
        # i.e. shift x-coordinate of the box corner to right or to left
        if extra_zoom == 'right_cam':
            bottomRight[0] += min(bottomRight[0]-upLeft[0], imgwidth-bottomRight[0])
        elif extra_zoom == 'left_cam':
            upLeft[0] -= min(upLeft[0], bottomRight[0]-upLeft[0])
        
        crops.append(cropBox(tmp_img, upLeft, bottomRight, target_height, target_width)[None,...])
        bboxes_zoomed.append(torch.cat((upLeft, bottomRight))[None,...])
    
    crops = torch.cat(crops, dim=0)
    bboxes_zoomed = torch.cat(bboxes_zoomed)
    
    return crops, bboxes_zoomed


def flip_v(x, cuda=True):
    x = flip(x.cpu().data)
    if cuda:
        x = x.cuda(non_blocking=True)
    with torch.no_grad():
        x = torch.autograd.Variable(x)
    return x


def flip(x):
    assert (x.dim() == 3 or x.dim() == 4)
    # dim = x.dim() - 1
    x = x.numpy().copy()
    if x.ndim == 3:
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(
                np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    # x = x.swapaxes(dim, 0)
    # x = x[::-1, ...]
    # x = x.swapaxes(0, dim)

    return torch.from_numpy(x.copy())


class Mscoco(data.Dataset):
    def __init__(
        self, 
        train=True, 
        sigma=1,
        scale_factor=(0.2, 0.3), 
        rot_factor=40, 
        label_type='Gaussian'
    ):
        self.img_folder = '/mnt/nfs/pose/AlphaPosePytorch/data/coco/images'    # root image folders
        self.is_train = train           # training set or test set
#         self.inputResH = opt.inputResH
#         self.inputResW = opt.inputResW
#         self.outputResH = opt.outputResH
#         self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 17
        self.nJoints_mpii = 16
        self.nJoints = 33

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


def shuffleLR(x, dataset):
    flipRef = dataset.flipRef
    assert (x.dim() == 3 or x.dim() == 4)
    for pair in flipRef:
        dim0, dim1 = pair
        dim0 -= 1
        dim1 -= 1
        if x.dim() == 4:
            tmp = x[:, dim1].clone()
            x[:, dim1] = x[:, dim0].clone()
            x[:, dim0] = tmp.clone()
            #x[:, dim0], x[:, dim1] = deepcopy((x[:, dim1], x[:, dim0]))
        else:
            tmp = x[dim1].clone()
            x[dim1] = x[dim0].clone()
            x[dim0] = tmp.clone()
            #x[dim0], x[dim1] = deepcopy((x[dim1], x[dim0]))
    return x


def shuffleLR_v(x, dataset, cuda=False):
    x = shuffleLR(x.cpu().data, dataset)
    if cuda:
        x = x.cuda(non_blocking=True)
    x = torch.autograd.Variable(x)
    return x


def transformBoxInvert_batch(pt, ul, br, inpH, inpW, resH, resW):
    '''
    pt:     [n, 17, 2]
    ul:     [n, 2]
    br:     [n, 2]
    '''
    center = (br - 1 - ul) / 2

    size = br - ul
    size[:, 0] *= (inpH / inpW)

    lenH, _ = torch.max(size, dim=1)   # [n,]transformBoxInvert_batch
    lenW = lenH * (inpW / inpH)

    _pt = (pt * lenH[:, np.newaxis, np.newaxis]) / resH
    _pt[:, :, 0] = _pt[:, :, 0] - ((lenW[:, np.newaxis].repeat(1, 17) - 1) /
                                   2 - center[:, 0].unsqueeze(-1).repeat(1, 17)).clamp(min=0)
    _pt[:, :, 1] = _pt[:, :, 1] - ((lenH[:, np.newaxis].repeat(1, 17) - 1) /
                                   2 - center[:, 1].unsqueeze(-1).repeat(1, 17)).clamp(min=0)

    new_point = torch.zeros(pt.size())
    new_point[:, :, 0] = _pt[:, :, 0] + ul[:, 0].unsqueeze(-1).repeat(1, 17)
    new_point[:, :, 1] = _pt[:, :, 1] + ul[:, 1].unsqueeze(-1).repeat(1, 17)
    return new_point

