import sys
import torch
import torch.nn as nn
from torchvision.models import resnet
from ..modules import Attention, FeatureExtraction, SpatialTransformer, BiLSTM


class Encoder(nn.Module):
    def __init__(self, in_feat: int = 1, nf: int = 20, im_size: tuple = (32, 100), 
                 resnet_version=18, pretrained=True, freeze_feature=True):
        super(Encoder, self).__init__()
        self.transformer = SpatialTransformer(nf=nf, img_size=im_size, imrec_size=im_size, img_channel=in_feat)
        self.feature = FeatureExtraction(in_channels=in_feat, version=resnet_version, pretrained=pretrained, freeze_base=freeze_feature)
        self.out_channels = self.feature.out_channels
        
    def forward(self, x):
        x = self.transformer(x)
        x = self.feature(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size: int, num_class: int, hidden_size: int = 256):
        super(Decoder, self).__init__()
        self.sequence = nn.Sequential(
            BiLSTM(input_size, hidden_size, hidden_size),
            BiLSTM(hidden_size, hidden_size, hidden_size)
        )
        self.attention = Attention(hidden_size, hidden_size, num_class, )

    def forward(self, feature: torch.Tensor, text, is_train=True, batch_max_length=25):
        contextual_feature = self.sequence(feature)
        contextual_feature = contextual_feature.contiguous()
        prediction = self.attention(contextual_feature, text, is_train, batch_max_length=batch_max_length)
        return prediction


class OCRNet(nn.Module):
    def __init__(self, num_class, in_feat: int = 1,
                 hidden_size: int = 256, nfid: int = 20, im_size: tuple = (32, 100),
                 resnet_version = 18, pretrained_feature=True, freeze_feature=True):
        super(OCRNet, self).__init__()
        self.encoder = Encoder(in_feat=in_feat, nf=nfid, im_size=im_size, 
                               resnet_version=resnet_version, pretrained=pretrained_feature, 
                               freeze_feature=freeze_feature)
        
        self.decoder = Decoder(input_size=self.encoder.out_channels, num_class=num_class, hidden_size=hidden_size)

    def forward(self, x: torch.Tensor, text, is_train=True, batch_max_length=25):
        features = self.encoder(x)
        prediction = self.decoder(features, text, is_train, batch_max_length)
        return prediction
    
    def read(self, x: torch.Tensor, converter, max_length=25):
        batch_size = x.size(0)
        used_device = x.get_device()
        if used_device == -1: used_device = 'cpu'
        texts_length = torch.IntTensor([max_length] * batch_size)
        texts_zeroes = torch.LongTensor(batch_size, max_length + 1).fill_(0)
        
        with torch.no_grad():
            features = self.encoder(x)
            prediction = self.decoder(features, texts_zeroes, is_train=False, batch_max_length=max_length)
            _, prediction_index = prediction.max(2)
            prediction_string = converter.decode(prediction_index, texts_length)
        
        return prediction_string, prediction_index
            


if __name__ == "__main__":
    # enc = Encoder()aqa
    test_data = torch.rand(3, 1, 224, 224)
    # out = enc(test_data)
    # print(out)

    num_class = 96
    im_size = (32, 100)
    model = OCRNet(num_class=num_class, im_size=im_size)
