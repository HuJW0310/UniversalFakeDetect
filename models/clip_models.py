from .clip import clip 
from PIL import Image
import torch.nn as nn


CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear( CHANNELS[name], num_classes )
 

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x) 
        # print(features.keys())
        """
        使用的是ViT-Large, 共24层
        选择第24、22、20层的[cls]feature做加权平均
        """
        if return_feature:
            return features['after_projection']
        # print(features['after_projection'].shape)
        # print(features['layer21'].shape)
        # print(features['layer19'].shape)
        # features = 0.5*features['after_projection'] + 0.3*features['layer21'] + 0.2*features['layer19']
        # print(features.shape)
        features = features['res_output']
        return self.fc(features)

