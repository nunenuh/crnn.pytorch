import string
import torch
from iqra.utils import AttnLabelConverter
import torchvision.transforms as VT
from iqra import transforms as NT
from iqra.models import crnn_v1
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader



class TextPredictor(object):
    def __init__(self, weight_path, device='cpu'):
        self.weight_path = weight_path
        self.device = device
        self._load_config()
        self._load_model()
        
    def _load_config(self):
        self.character = string.printable[:-6]
        self.converter = AttnLabelConverter(self.character)
        self.num_class = len(self.converter.character)
        self.batch_max_length = 25
        self.img_size = (32, 100)
        
        
    def _load_model(self):
        state_dict = torch.load(self.weight_path, map_location=torch.device(self.device))
        self.model = crnn_v1.OCRNet(num_class=self.num_class, im_size=self.img_size, hidden_size=256)
        self.model.load_state_dict(state_dict)
        
    def _predict(self, images:list):
        dloader = self._transform_loader(images)
        images = next(iter(dloader))
        batch_size = images.shape[0]
        
        
        length = torch.IntTensor([self.batch_max_length] * batch_size)
        preds = self.model(images)
        preds = preds[:, :self.batch_max_length, :]
        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index, length)
        preds_clean = self._clean_prediction(preds_str)    
        
        return preds_clean
    
    def predict(self, images):
        return self._predict(images)
    
    def _clean_prediction(self, preds_str):
        out = []
        for prd_st in preds_str:
            word = prd_st.split("[s]")[0]
            out.append(word)
        return out
    
    def _transform_loader(self, images):
        transform = VT.Compose([
            VT.ToPILImage(),
            VT.Grayscale(),
            NT.ResizeRatioWithRightPad(size=self.img_size),
            VT.ToTensor(),
            VT.Normalize(mean=(0.5,), std=(0.5,))
        ])
        
        out = []
        for image in images:
            timg = transform(image)
            out.append(timg)
            
        return DataLoader(out, batch_size=len(out))
    
        