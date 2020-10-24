import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Word Definition:
    Fiducial is (especially of a point or line) assumed as a fixed basis of comparison.
    STN is Spatial Transformer Network
    TPS is Thin Plate Spline 


Formula:
    C = set of fiducial point
    C'= base fiducial points
    P = the grid generator generates a sampling grid
    T = Transformation point
    I' = Rectified Image 
"""

class LocalizationNetwork(nn.Module):
    def __init__(self, nf, img_channel):
        super(LocalizationNetwork, self).__init__()
        self.nf = nf
        self.img_channel = img_channel
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=self.img_channel, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # batch_size x 64 x I_height/2 x I_width/2
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # batch_size x 128 x I_height/4 x I_width/4
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # batch_size x 256 x I_height/8 x I_width/8
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # batch_size x 512
        )
        
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True)
        )
        
        self.fc2 = nn.Linear(256, self.nf * 2)
        
        self._init_weight_bias_data_fc2(self.nf)
        
   
    def _init_weight_bias_data_fc2(self, nf: int):
        """see RARE paper Fig. 6 (a)
       
        definition : 
            cp = control points

        Args:
            nf([type]): [description]
        """
        
        hnf = nf // 2
        cp_x = torch.linspace(-1.0, 1.0, steps=hnf)
        cp_y_top = torch.linspace(0.0, -1.0, steps=hnf)
        cp_y_bottom = torch.linspace(1.0, 0.0, steps=hnf)
        
        cp_top = torch.stack([cp_x, cp_y_top], dim=1)
        cp_bottom = torch.stack([cp_x, cp_y_bottom], dim=1)
        initial_bias = torch.cat([cp_top, cp_bottom], dim=0)
        
        #fill bias data with precalculated fiducial point
        self.fc2.bias.data = initial_bias.float().view(-1) 
        
        #fill weight data with zero value
        self.fc2.weight.data.fill_(0) 
        
        
    def forward(self, x):
        batch_size = x.size(0)
        features = self.features(x)
        features = features.view(batch_size, -1)
        
        fc1 = self.fc1(features)
        fc2 = self.fc2(fc1)
        predicted_coordinate = fc2.view(batch_size, self.nf, 2)
        
        return predicted_coordinate
    

class GridGenerator(nn.Module):
    
    def __init__(self, nf: int, imrec_size: tuple, eps: float = 1e-6):
        """[summary]

        Args:
            nf (int): [description]
            imrec_size (tuple): image rectified size
            eps (float, optional): [description]. Defaults to 1e-6.
        """
        super(GridGenerator, self).__init__()
        self.nf = nf
        self.imrec_size = imrec_size
        self.ir_height, self.ir_width = imrec_size
        self.eps = eps
        
        self.c = self._create_c(self.nf)
        self.p = self._create_p(self.ir_width, self.ir_height)
        
        # self.inv_delta_c = self._inv_delta_c(self.nf, self.c)
        # self.p_hat = self._p_hat(self.nf, self.c, self.p)
        
        inv_delta_c = self._create_inv_delta_c(self.nf, self.c)
        p_hat = self._create_p_hat(self.nf, self.c, self.p)
        
        self.register_buffer("inv_delta_c", inv_delta_c)
        self.register_buffer("p_hat", p_hat)

    def _create_c(self, nf: int):
        hnf = nf // 2
        
        cp_x = torch.linspace(-1.0, 1.0, steps=hnf)
        cp_y_top = -1 * torch.ones(hnf)
        cp_y_bottom = torch.ones(hnf)
        cp_top = torch.stack([cp_x, cp_y_top], dim=1)
        cp_bottom = torch.stack([cp_x, cp_y_bottom], dim=1)
        c = torch.cat([cp_top, cp_bottom], dim=0)
        
        return c  # F x 2   
    
    def _create_p(self, ir_witdh: int, ir_height: int):
        """[summary]

        Args:
            ir_witdh ([int]): image rectified width
            ir_height ([int]): image rectified height
        """
        
        ir_grid_x = (torch.arange(-ir_witdh, ir_witdh, 2) + 1.0) / ir_witdh
        ir_grid_y = (torch.arange(-ir_height, ir_height, 2) + 1.0) / ir_height
        p = torch.stack(torch.meshgrid(ir_grid_x, ir_grid_y), dim=2)
        p = p.view(-1, 2)
        return p  
    
    def _create_p_hat(self, nf: int, c: torch.tensor, p: torch.tensor):
        """[summary]

        Args:
            nf (int): number of fiducial points denote by F
            c (torch.tensor): set of fiducial points denote by C
            p (torch.tensor): sampling grid denote by P

        Returns:
            (torch.tensor): [description]
        """
        
        n = p.size(0) # n (= self.ir_witdh x self.ir_height)
        # print(n)
        p_tile = p.unsqueeze(dim=1).repeat((1, nf, 1)) # n x 2 -> n x 1 x 2 -> n x F x 2
        c_tile = c.unsqueeze(dim=0) # 1 x F x 2
        p_diff = p_tile - c_tile # n x F x 2
        rbf_norm = torch.norm(p_diff, p=2, dim=2, keepdim=False)
        rbf = torch.mul(torch.square(rbf_norm), torch.log(rbf_norm + self.eps)) # n x F
        p_hat = torch.cat([torch.ones(n,1), p, rbf], dim=1)
        
        return p_hat.float()
    
    def _create_inv_delta_c(self, nf: int, c):
        """[summary]

        Args:
            nf ([type]): [description]
            c ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        c_hat = torch.zeros((nf, nf)).float()
        for i in range(0, nf):
            for j in range(0, nf):
                r = torch.norm(c[i] - c[j])
                c_hat[i, j] = r
                c_hat[j, i] = r
                
        torch.diagonal(c_hat).fill_(1) 
        c_hat = (c_hat ** 2) * torch.log(c_hat)
        
        delta_c = torch.cat([
            torch.cat([torch.ones((nf, 1)), c, c_hat], dim=1),
            torch.cat([torch.zeros((2,3)), c.transpose(0,1)], dim=1),
            torch.cat([torch.zeros((1,3)), torch.ones((1, nf))], dim=1)
        ])

        inv_delta_c = torch.inverse(delta_c)

        return inv_delta_c
    
    
    def forward(self, p_prime):
        """ Generate Grid from data [batch_size x F x 2] """
        batch_size = p_prime.size(0)
        inv_delta_c = self.inv_delta_c.repeat(batch_size, 1, 1)
        p_hat = self.p_hat.repeat(batch_size, 1, 1)
        
        zeroes = torch.zeros(batch_size, 3, 2).float()
        device = p_prime.get_device()
        if device == -1: device='cpu'
        zeroes = zeroes.to(device)
        
        c_prime_zeros = torch.cat([p_prime, zeroes], dim=1)
        transformation = torch.bmm(inv_delta_c, c_prime_zeros)
        p_prime = torch.bmm(p_hat, transformation)
        
        return p_prime
    
    
class SpatialTransformer(nn.Module):
    """ Rectification Network of RARE, namely TPS based STN """
    
    def __init__(self, nf, img_size, imrec_size, img_channel=1):
        super(SpatialTransformer, self).__init__()
        
        """ Based on RARE TPS
        input:
            batch_I: Batch Input Image [batch_size x img_channel_num x I_height x I_width]
            img_size : (height, width) of the input image I
            imrec_size : (height, width) of the rectified image I_r
            img_channel_num : the number of channels of the input image I
        output:
            batch_I_r: rectified image [batch_size x img_channel_num x I_r_height x I_r_width]
        """
        
        self.nf = nf
        self.img_size = img_size
        self.imrec_size = imrec_size # = (I_r_height, I_r_width)
        self.ir_height, self.ir_width = self.imrec_size
        self.img_channel_num = img_channel
        
        self.localization_network = LocalizationNetwork(self.nf, self.img_channel_num)
        self.grid_generator = GridGenerator(self.nf, self.imrec_size)
        
        
    def forward(self, x):
        c_prime = self.localization_network(x) # batch_size x K x 2
        p_prime = self.grid_generator(c_prime)  # batch_size x n (= I_r_width x I_r_height) x 2
        p_prime = p_prime.view([p_prime.size(0), self.imrec_size[0], self.imrec_size[1], 2])
        x = F.grid_sample(x, p_prime, padding_mode='border', align_corners=True)

        return x
        
        

# create test code for stn        
if __name__ == "__main__":
    nf = 20
    im_size = (24,24)
    input_channel = 1
   
    test_data = torch.rand(2, 1, 16, 16)
    
    
    # locnet = LocalizationNetwork(nf, input_channel)
    # print('LocalizationNetwork',locnet)
    # out_locnet = locnet(test_data)
    # print('out_locnet',out_locnet)
    
    # gridgen = GridGenerator(nf, im_size)
    # # print('fiducial_points',gridgen.fiducial_points)
    # # print('p',gridgen.p)
    # # print('p_hat', gridgen.p_hat)
    # # print('inv_delta', gridgen.inv_delta_fiducial_points)
    
    # out_gridgen = gridgen(out_locnet)
    # print(out_gridgen)
    
    
    STN = SpatialTransformerNetwork(
        nf=nf, img_size=im_size, 
        imrec_size=im_size, img_channel=input_channel
    )
    print(STN)
    
    out_stn = STN(test_data)
    print('out_stn', out_stn)
    print('out_stn', out_stn.shape)
    
    