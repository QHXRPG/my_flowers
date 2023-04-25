import torch
from torchvision.models import resnet50
import torch.nn as nn
from torchvision.models import vgg16

model = resnet50(pretrained=True)
vgg_model = vgg16(pretrained=True)

def to_tensor(img):
    img = torch.tensor(img, dtype=torch.float32)
    img = img.permute(2, 0, 1).contiguous()
    img = img.unsqueeze(0)
    return img

class Classifies_model(nn.Module):
    def __init__(self):
        super(Classifies_model, self).__init__()
        self.main_model = nn.Sequential(*list(model.children())[:-1])
        self.classifies_layer = nn.Sequential(nn.Linear(2048,1024),
                                              nn.ReLU(),
                                              nn.Linear(1024,256),
                                              nn.ReLU(),
                                              nn.Linear(256,64),
                                              nn.ReLU(),
                                              nn.Linear(64,4))
    def forward(self,input):
        feature_map = self.main_model(input)
        feature_map = feature_map.view(feature_map.shape[0],-1)
        output = self.classifies_layer(feature_map)
        return output

class Vgg16_Classifies_model(nn.Module):
    def __init__(self):
        super(Vgg16_Classifies_model, self).__init__()
        self.main_model = nn.Sequential(*list(vgg_model.children())[:-2])
        self.apt_pool = nn.AdaptiveAvgPool2d(1)
        self.classifies_layer = nn.Sequential(nn.Linear(512,1024),
                                              nn.ReLU(),
                                              nn.Linear(1024,256),
                                              nn.ReLU(),
                                              nn.Linear(256,64),
                                              nn.ReLU(),
                                              nn.Linear(64,4))

    def forward(self, input):
        feature_map = self.main_model(input)
        feature_map = self.apt_pool(feature_map)
        feature_map = feature_map.view(feature_map.shape[0], -1)
        output = self.classifies_layer(feature_map)
        return output
