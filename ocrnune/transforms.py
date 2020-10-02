import math
from typing import *

import torch
import torchvision.transforms.functional as VF
import torchvision.transforms as VT
from . import functional as NF

import PIL
from PIL import Image



class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = VF.to_tensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class ImagePad(object):

    def __init__(self, max_size=(32, 100)):
        self.max_size = max_size
        self.max_height, self.max_width = max_size
        self.max_width_half = math.floor(self.max_width / 2)

    def __call__(self, img: Image):
        pad_img = NF.right_pad(img, self.max_size)
        return pad_img
    
    
class ResizeRatioWithRightPad(object):
    def __init__(self, size=(32, 100), interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img: Image):
        img_resized = NF.resize_with_keep_ratio(img, self.size, self.interpolation)
        img_padded = NF.right_pad(img_resized, max_size=self.size)
        return img_padded
    
            

