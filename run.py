import os
import sys
sys.path.append(os.getcwd())

import torch
from iqra.models.crnn import *



if __name__ == '__main__':
    image_data = torch.rand(3,1,224,224)
    text_data = torch.rand(3,512)
    # out = enc(test_data)
    # print(out)


    num_class = 96
    im_size = (32, 100)
    model = OCRNet(num_class = num_class, im_size=im_size)
    out = model(image_data, text_data)
    print(out)
    print(out.shape)
    