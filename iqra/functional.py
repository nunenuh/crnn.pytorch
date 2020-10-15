import math
import torch

import PIL
from PIL import Image

import torchvision.transforms.functional as VF


def resize_with_keep_ratio(img: Image, size: tuple = (32, 100), interpolation=Image.BICUBIC):
    """[summary]

    Args:
        img (Image): [description]
        size (tuple, optional): [description]. Defaults to (32, 100).
        interpolation ([type], optional): [description]. Defaults to Image.BICUBIC.

    Returns:
        [type]: [description]
    """
    
    rz_h, rz_w = size
    im_w, im_h = img.size
    ratio = im_w / float(im_h)

    if math.ceil(rz_h * ratio) > rz_w:
        new_w = rz_w
    else:
        new_w = math.ceil(rz_h * ratio)

    resized = img.resize((new_w, rz_h), interpolation)

    return resized


def right_pad(img: Image, max_size=(32,100)):
    img = VF.to_tensor(img)
    im_c, im_h, im_w = img.size()
    mh, mw = max_size
    
    chw = (im_c, mh, mw)
    pad_img = torch.FloatTensor(*chw).fill_(0)
    pad_img[:, :, :im_w] = img  # right pad
    if mw != im_w:  # add border Pad
        img = img[:, :, im_w - 1].unsqueeze(2)
        pad_img[:, :, im_w:] = img.expand(im_c, im_h, mw - im_w)

    pad_img = VF.to_pil_image(pad_img.squeeze_(0))
    return pad_img
