import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


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
    def __init__(self, num_fiducial, img_channel_num):
        super(LocalizationNetwork, self).__init__()
        self.num_fiducial = num_fiducial
        self.img_channel_num = img_channel_num
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=self.img_channel_num, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
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
        
        self.fc2 = nn.Linear(256, self.num_fiducial * 2)
        
        self._init_weight_bias_data_fc2(self.num_fiducial)
        
   
    def _init_weight_bias_data_fc2(self, num_fiducial: int):
        """see RARE paper Fig. 6 (a)
       
        definition : 
            ctrl_pts = control points

        Args:
            num_fiducial ([type]): [description]
        """
        
        half_num_fiducial = num_fiducial // 2
        ctrl_pts_x = torch.linspace(-1.0, 1.0, steps=half_num_fiducial)
        ctrl_pts_y_top = torch.linspace(0.0, -1.0, steps=half_num_fiducial)
        ctrl_pts_y_bottom = torch.linspace(1.0, 0.0, steps=half_num_fiducial)
        
        ctrl_pts_top = torch.stack([ctrl_pts_x, ctrl_pts_y_top], dim=1)
        ctrl_pts_bottom = torch.stack([ctrl_pts_x, ctrl_pts_y_bottom], dim=1)
        initial_bias = torch.cat([ctrl_pts_top, ctrl_pts_bottom], dim=0)
        
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
        predicted_coordinate = fc2.view(batch_size, self.num_fiducial, 2)
        
        return predicted_coordinate
    

class GridGenerator(nn.Module):
    
    def __init__(self, num_fiducial: int, ir_size: tuple, eps: float = 1e-6):
        """[summary]

        Args:
            num_fiducial (int): [description]
            ir_size (tuple): image rectified size
            eps (float, optional): [description]. Defaults to 1e-6.
        """
        super(GridGenerator, self).__init__()
        self.num_fiducial = num_fiducial
        self.ir_size = ir_size
        self.ir_height, self.ir_width = ir_size
        self.eps = eps
        
        self.fiducial_points = self._build_fiducial_points(self.num_fiducial)
        self.sampling_grid = self._build_sampling_grid(self.ir_width, self.ir_height)
        
        # self.inverse_delta_fiducial_points = self._build_inverse_delta_fiducial_points(self.num_fiducial, self.fiducial_points)
        # self.sampling_grid_hat = self._build_sampling_grid_hat(self.num_fiducial, self.fiducial_points, self.sampling_grid)
        
        inverse_delta_fiducial_points = self._build_inverse_delta_fiducial_points(self.num_fiducial, self.fiducial_points)
        sampling_grid_hat = self._build_sampling_grid_hat(self.num_fiducial, self.fiducial_points, self.sampling_grid)
        
        self.register_buffer("inverse_delta_fiducial_points", inverse_delta_fiducial_points)
        self.register_buffer("sampling_grid_hat", sampling_grid_hat)

    def _build_fiducial_points(self, num_fiducial: int):
        half_num_fiducial = num_fiducial // 2
        
        ctrl_pts_x = torch.linspace(-1.0, 1.0, steps=half_num_fiducial)
        ctrl_pts_y_top = -1 * torch.ones(half_num_fiducial)
        ctrl_pts_y_bottom = torch.ones(half_num_fiducial)
        ctrl_pts_top = torch.stack([ctrl_pts_x, ctrl_pts_y_top], dim=1)
        ctrl_pts_bottom = torch.stack([ctrl_pts_x, ctrl_pts_y_bottom], dim=1)
        fiducial_points = torch.cat([ctrl_pts_top, ctrl_pts_bottom], dim=0)
        
        return fiducial_points  # F x 2   
    
    def _build_sampling_grid(self, ir_witdh: int, ir_height: int):
        """[summary]

        Args:
            ir_witdh ([int]): image rectified width
            ir_height ([int]): image rectified height
        """
        
        ir_grid_x = (torch.arange(-ir_witdh, ir_witdh, 2) + 1.0) / ir_witdh
        ir_grid_y = (torch.arange(-ir_height, ir_height, 2) + 1.0) / ir_height
        sampling_grid = torch.stack(torch.meshgrid(ir_grid_x, ir_grid_y), dim=2)
        sampling_grid = sampling_grid.view(-1, 2)
        return sampling_grid  
    
    def _build_sampling_grid_hat(self, num_fiducial: int, fiducial_points: torch.tensor, sampling_grid: torch.tensor):
        """[summary]

        Args:
            num_fiducial (int): number of fiducial points denote by F
            fiducial_points (torch.tensor): set of fiducial points denote by C
            sampling_grid (torch.tensor): sampling grid denote by P

        Returns:
            (torch.tensor): [description]
        """
        
        n = sampling_grid.size(0) # n (= self.ir_witdh x self.ir_height)
        # print(n)
        sampling_grid_tile = sampling_grid.unsqueeze(dim=1).repeat((1, num_fiducial, 1)) # n x 2 -> n x 1 x 2 -> n x F x 2
        fiducial_points_tile = fiducial_points.unsqueeze(dim=0) # 1 x F x 2
        sampling_grid_diff = sampling_grid_tile - fiducial_points_tile # n x F x 2
        rbf_norm = torch.norm(sampling_grid_diff, p=2, dim=2, keepdim=False)
        rbf = torch.mul(torch.square(rbf_norm), torch.log(rbf_norm + self.eps)) # n x F
        sampling_grid_hat = torch.cat([torch.ones(n,1), sampling_grid, rbf], dim=1)
        
        return sampling_grid_hat.float()
    
    def _build_inverse_delta_fiducial_points(self, num_fiducial: int, fiducial_points):
        """[summary]

        Args:
            num_fiducial ([type]): [description]
            fiducial_points ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        fiducial_points_hat = torch.zeros((num_fiducial, num_fiducial)).float()
        for i in range(0, num_fiducial):
            for j in range(0, num_fiducial):
                r = torch.norm(fiducial_points[i] - fiducial_points[j])
                # print(r)
                fiducial_points_hat[i, j] = r
                fiducial_points_hat[j, i] = r
                
        torch.diagonal(fiducial_points_hat).fill_(1) 
        fiducial_points_hat = (fiducial_points_hat ** 2) * torch.log(fiducial_points_hat)
        
        fiducial_points_delta = torch.cat([
            torch.cat([torch.ones((num_fiducial, 1)), fiducial_points, fiducial_points_hat], dim=1),
            torch.cat([torch.zeros((2,3)), fiducial_points.transpose(0,1)], dim=1),
            torch.cat([torch.zeros((1,3)), torch.ones((1, num_fiducial))], dim=1)
        ])
        # print('fiducial_points_delta', fiducial_points_delta.isnan())
        
        
        inverse_delta_fiducial_points = torch.inverse(fiducial_points_delta)
        # print('inverse_delta_fiducial_points', inverse_delta_fiducial_points)
        
        
        return inverse_delta_fiducial_points
    
    
    def forward(self, batch_sampling_grid_prime):
        """ Generate Grid from data [batch_size x F x 2] """
        batch_size = batch_sampling_grid_prime.size(0)
        batch_inv_delta_fiducial_points = self.inverse_delta_fiducial_points.repeat(batch_size, 1, 1)
        batch_sampling_grid_hat = self.sampling_grid_hat.repeat(batch_size, 1, 1)
        
        # zeroes = torch.zeros(batch_size, 3, 2).float().to(device)
        zeroes = torch.zeros(batch_size, 3, 2).float()
        batch_fiducial_points_prime_with_zeros = torch.cat([batch_sampling_grid_prime, zeroes], dim=1)
        batch_transformation = torch.bmm(batch_inv_delta_fiducial_points, batch_fiducial_points_prime_with_zeros)
        batch_sampling_grid_prime = torch.bmm(batch_sampling_grid_hat, batch_transformation)
        
        return batch_sampling_grid_prime
    
    

class SpatialTransformerNetwork(nn.Module):
    """ Rectification Network of RARE, namely TPS based STN """
    
    def __init__(self, num_fiducial, img_size, img_rectified_size, img_channel_num=1):
        super(SpatialTransformerNetwork, self).__init__()
        
        """ Based on RARE TPS
        input:
            batch_I: Batch Input Image [batch_size x img_channel_num x I_height x I_width]
            img_size : (height, width) of the input image I
            img_rectified_size : (height, width) of the rectified image I_r
            img_channel_num : the number of channels of the input image I
        output:
            batch_I_r: rectified image [batch_size x img_channel_num x I_r_height x I_r_width]
        """
        
        self.num_fiducial = num_fiducial
        self.img_size = img_size
        self.img_rectified_size = img_rectified_size # = (I_r_height, I_r_width)
        self.ir_height, self.ir_width = self.img_rectified_size
        self.img_channel_num = img_channel_num
        
        self.localization_network = LocalizationNetwork(self.num_fiducial, self.img_channel_num)
        self.grid_generator = GridGenerator(self.num_fiducial, self.img_rectified_size)
        
        
    def forward(self, x):
        fiducial_points_prime = self.localization_network(x) # batch_size x K x 2
        sampling_grid_prime = self.grid_generator(fiducial_points_prime)  # batch_size x n (= I_r_width x I_r_height) x 2
        sampling_grid_prime = sampling_grid_prime.view([sampling_grid_prime.size(0), self.img_rectified_size[0], self.img_rectified_size[1], 2])
        
        x = F.grid_sample(x, sampling_grid_prime, padding_mode='border', align_corners=True)

        return x
        
        

# create test code for stn        
if __name__ == "__main__":
    nfid = 20
    im_size = (24,24)
    input_channel = 1
   
    test_data = torch.rand(2, 1, 16, 16)
    
    
    # locnet = LocalizationNetwork(nfid, input_channel)
    # print('LocalizationNetwork',locnet)
    # out_locnet = locnet(test_data)
    # print('out_locnet',out_locnet)
    
    # gridgen = GridGenerator(nfid, im_size)
    # # print('fiducial_points',gridgen.fiducial_points)
    # # print('sampling_grid',gridgen.sampling_grid)
    # # print('sampling_grid_hat', gridgen.sampling_grid_hat)
    # # print('inv_delta', gridgen.inverse_delta_fiducial_points)
    
    # out_gridgen = gridgen(out_locnet)
    # print(out_gridgen)
    
    
    # STN = SpatialTransformerNetworkTPS(
    #     num_fiducial=nfid, img_size=im_size, 
    #     img_rectified_size=im_size, img_channel_num=input_channel
    # )
    # print(STN)
    
    # out_stn = STN(test_data)
    # print('out_stn', out_stn)
    # print('out_stn', out_stn.shape)
    
    