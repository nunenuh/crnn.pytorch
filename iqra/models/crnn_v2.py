import sys
import torch
import torch.nn as nn
from torchvision.models import resnet
from ..modules import Attention, FeatureExtractor, BiLSTM
from ..modules.spatial import SpatialTransformer
from ..modules.transformer import Transformer
from ..ops import net

class Encoder(nn.Module):
    def __init__(self, in_feat: int = 1, out_feat=512, nf: int = 20, im_size: tuple = (32, 100)):
        super(Encoder, self).__init__()
        self.spatial_transformer = SpatialTransformer(nf=nf, img_size=im_size, imrec_size=im_size, img_channel=in_feat)
        self.feature_extractor = FeatureExtractor(in_channels=in_feat, out_channels=out_feat)
        self.out_channels = out_feat
        
    def forward(self, x):
        x = self.spatial_transformer(x)
        x = self.feature_extractor(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size: int, num_class: int, hidden_size: int = 256):
        super(Decoder, self).__init__()
        self.sequence = nn.Sequential(
            BiLSTM(input_size, hidden_size, hidden_size),
            BiLSTM(hidden_size, hidden_size, hidden_size)
        )
        self.prediction = Transformer(hidden_size, num_class)

    def forward(self, feature: torch.Tensor):
        contextual_feature = self.sequence(feature)
        contextual_feature = contextual_feature.contiguous()
        prediction = self.prediction(contextual_feature)
        return prediction


class TransformerOCRNet(nn.Module):
    def __init__(self, num_class, in_feat: int = 1, out_feat=512, hidden_size: int = 256, 
                 nfid: int = 20, im_size: tuple = (32, 100)):
        super(TransformerOCRNet, self).__init__()
        self.encoder = Encoder(in_feat=in_feat, out_feat=out_feat, nf=nfid, im_size=im_size)
        self.decoder = Decoder(input_size=out_feat, num_class=num_class, hidden_size=hidden_size)

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        prediction = self.decoder(features)
        return prediction
    
    def freeze_spatial(self):
        net.freeze(self.encoder.spatial_transformer)
        
    def freeze_feature(self):
        net.freeze(self.encoder.feature_extractor)
    
    def freeze_encoder(self):
        net.freeze(self.encoder)
            
    def freeze_sequence(self):
        net.freeze(self.decoder.sequence)
            
    def freeze_prediction(self):
        net.freeze(self.decoder.prediction)


if __name__ == "__main__":
    # enc = Encoder()aqa
    test_data = torch.rand(3, 1, 224, 224)
    # out = enc(test_data)
    # print(out)

    num_class = 96
    im_size = (32, 100)
    model = TransformerOCRNet(num_class=num_class, im_size=im_size)
