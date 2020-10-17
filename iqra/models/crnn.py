import sys
import torch
import torch.nn as nn
from ..modules import Attention, FeatureExtraction, SpatialTransformerNetwork as STN, BiLSTM


class Encoder(nn.Module):
    def __init__(self, in_feat: int = 1, out_feat: int = 512,
                 nf: int = 20, im_size: tuple = (32, 100)):
        super(Encoder, self).__init__()
        self.stn = STN(nf=nf, img_size=im_size, imrec_size=im_size, img_channel=in_feat)

        self.feature = FeatureExtraction(in_feat=in_feat, out_feat=out_feat)

    def forward(self, x):
        x = self.stn(x)
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
    def __init__(self, num_class, in_feat: int = 1, out_feat: int = 512,
                 hidden_size: int = 256, nfid: int = 20, im_size: tuple = (32, 100)):
        super(OCRNet, self).__init__()
        self.encoder = Encoder(in_feat=in_feat, out_feat=out_feat, nf=nfid, im_size=im_size)
        self.decoder = Decoder(input_size=out_feat, num_class=num_class, hidden_size=hidden_size)

    def forward(self, x: torch.Tensor, text, is_train=True, batch_max_length=25):
        features = self.encoder(x)
        prediction = self.decoder(features, text, is_train, batch_max_length)
        return prediction


if __name__ == "__main__":
    # enc = Encoder()aqa
    test_data = torch.rand(3, 1, 224, 224)
    # out = enc(test_data)
    # print(out)

    num_class = 96
    im_size = (32, 100)
    model = OCRNet(num_class=num_class, im_size=im_size)
