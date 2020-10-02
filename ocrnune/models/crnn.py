import torch
import torch.nn as nn
from ..modules import attention, feature, prediction, sequence, transformation 


class EncoderOCR(nn.Module):
    def __init__(self, in_feat: int = 1, out_feat: int = 512, 
                 num_fiducial: int = 20, im_size: tuple = (32, 100)):
        super(EncoderOCR, self).__init__()
        self.transformer = transformation.SpatialTransformerNetwork(num_fiducial=num_fiducial, 
                                                                    img_size=im_size, img_rectified_size=im_size, 
                                                                    img_channel_num=in_feat)
        
        self.feature_extraction = feature.ResNetFeatureExtraction(in_feat=in_feat, out_feat=out_feat)
        # self.pool = nn.AdaptiveAvgPool2d((None, 1))
        
        
    def forward(self, x):
        x = self.transformer(x)
        x = self.feature_extraction(x)
        return x
    
    
class DecoderOCR(nn.Module):
    def __init__(self, input_size: int, num_class: int, hidden_size: int = 256):
        super(DecoderOCR, self).__init__()
        self.sequence = nn.Sequential(
            sequence.BiLSTM(input_size, hidden_size, hidden_size),
            sequence.BiLSTM(hidden_size, hidden_size, hidden_size)
        )
        self.prediction = attention.Attention(hidden_size, hidden_size, num_class)
        
        
    def forward(self, feature: torch.Tensor, text, is_train=True, batch_max_length=25):
        contextual_feature = self.sequence(feature)
        prediction = self.prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=batch_max_length)
        return prediction
    

class OCR(nn.Module):
    def __init__(self, num_class, in_feat: int = 1, out_feat: int = 512, 
                 hidden_size: int = 256, nfid: int = 20, im_size: tuple = (32, 100)):
        super(OCR, self).__init__()
        self.encoder = EncoderOCR(in_feat=in_feat, out_feat=out_feat, num_fiducial=nfid, im_size=im_size)
        self.decoder = DecoderOCR(input_size=out_feat, num_class=num_class, hidden_size=hidden_size)
        
    def forward(self, x: torch.Tensor, text, is_train=True, batch_max_length=25):
        features = self.encoder(x)
        prediction = self.decoder(features, text, is_train, batch_max_length)
        return prediction
    
    
    
if __name__ == "__main__":
    # enc = EncoderOCR
    # test_data = torch.rand(3,1,224,224)
    # out = enc(test_data)
    # print(out)
    num_class = 96
    im_size = (32, 100)
    model = OCR(num_class = num_class, im_size=im_size)
    