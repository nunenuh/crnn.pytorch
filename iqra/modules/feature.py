import torch
import torch.nn as nn
import torchvision.models.resnet as resnet


class FeatureExtraction(nn.Module):
    def __init__(self, in_feat, out_feat, resnet_block = resnet.BasicBlock, layers=[3, 4, 6, 3], freeze_network=False):
        super(FeatureExtraction, self).__init__()    
        self.resnet_base = ResNetBase(resnet_block, layers, in_channels=in_feat, out_channels=out_feat)
        
        if freeze_network:
            self._freeze_network()
        
    
    def _freeze_network(self):
        for param in self.resnet_base.parameters():
            param.requires_grad_(False)
        
    def forward(self, x):
        x = self.resnet_base(x)
        return x


class ResNetBase(nn.Module):
    
    def __init__(self, block, layers, in_channels=1, out_channels=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetBase, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.zero_init_residual = zero_init_residual
        
        self.out_channels = [out_channels // 4, out_channels // 2, out_channels, out_channels]

        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
       
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.out_channels[0], layers[0])
        self.layer2 = self._make_layer(block, self.out_channels[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, self.out_channels[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, self.out_channels[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((None, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()
        self._init_zero_init_residual()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
                
    def _init_zero_init_residual(self):
         # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, resnet.BasicBlock):
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
                resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = x.permute(0, 3, 1, 2)
        x = self.avgpool(x)
        x = x.squeeze(3)
        return x



# def _resnet_base(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNetFeatureBase(block, layers, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model

if __name__ == "__main__":
    data = torch.rand((2,1,224,224))
    model = ResNetFeatureExtraction(in_feat=1, out_feat=128)
    # rnf = resnet.resnet34(pretrained=True)
    print(model)
    out = model(data)
    print(out.shape)
    # print(nfor.shape)
    # print(output.shape)
    # model  = ResNetFeatureBase(resnet.BasicBlock, [3, 4, 6, 3], in_channels=1, out_channels=512)
    # output = model(test_data)
    # print(model)
    # print(output.shape)