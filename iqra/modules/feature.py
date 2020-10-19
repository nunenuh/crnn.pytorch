import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from torchvision.models import resnet
from torchvision.models.resnet import model_urls
import torch.utils.model_zoo as model_zoo


class FeatureExtraction(nn.Module):
    def __init__(self, in_channels, version=18, pretrained=True, freeze_base=True):
        super(FeatureExtraction, self).__init__()    
        self.feature = _build_resnet(version=version, pretrained=pretrained, freeze=freeze_base)
        self.feature.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.avgpool =  nn.AdaptiveAvgPool2d((None, 1))
        
        self.out_channels = self.feature.last_channels
        
    def forward(self, x):
        x = self.feature(x)
        
        # print(f'feature: {x.shape}')
        
        x = x.permute(0, 3, 1, 2)
        x = self.avgpool(x)
        # print(f'avgpool: {x.shape}')
        x = x.squeeze(3)
        return x

    
class ResNetFeatureBase(resnet.ResNet):
    def __init__(self, block, layers):
        super(ResNetFeatureBase, self).__init__(block, layers)
        self.expansion = block.expansion
        self.last_channels = block.expansion * 512

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        

        return x
    

resnet_block_config = {
    '18': [2, 2, 2, 2], '34': [3, 4, 6, 3],
    '50': [3, 4, 6, 3], '101': [3, 4, 23, 3],
    '152': [3, 8, 36, 3]
}       

def _build_resnet(version=18, pretrained=True, freeze=True):
    ver = version
    block = resnet_block_config
    name_ver = 'resnet'+str(ver)
    if not str(ver) in block.keys():
        raise NotImplementedError(f'resnet version {ver} is not Implemented yet!')
    
    if ver>=50:
        model = ResNetFeatureBase(resnet.Bottleneck, block[str(ver)])
    else:
        model = ResNetFeatureBase(resnet.BasicBlock, block[str(ver)])
    
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[name_ver]))
        
    del model.avgpool  
    del model.fc
    
    if freeze:
        for param in model.parameters():
            param.requires_grad_(False)
    return model


    
if __name__ == "__main__":
    data = torch.rand((2,1,224,224))
    model = FeatureExtraction(in_feat=1, out_feat=128, version=32)
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