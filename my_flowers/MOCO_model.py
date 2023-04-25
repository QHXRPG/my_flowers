import torch
from torchvision.models import resnet50
import torch.nn as nn

model = resnet50(pretrained=True)

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.l1 = nn.Sequential(*list(model.children())[:-1])
        self.l2 = nn.Sequential(nn.Linear(2048,1000))

    def forward(self, x):
        x = self.l1(x)
        B, _, _, _ = x.shape
        output = x.view(B,-1)
        output = self.l2(output)
        return output

MOCO_pretrain_model = encoder()
MOCO_pretrain_model = torch.nn.DataParallel(MOCO_pretrain_model,device_ids=[0])
MOCO_pretrain_model.load_state_dict(torch.load('/mnt/f_q.pt'))
MOCO_pretrain_model = MOCO_pretrain_model.cuda()


class MOCO_Classifies_model(nn.Module):
    def __init__(self):
        super(MOCO_Classifies_model, self).__init__()
        self.main_model = MOCO_pretrain_model
        self.classifies_layer = nn.Sequential(nn.Linear(1000,40),
                                              nn.ReLU(),
                                              nn.Linear(40,4))
    def forward(self,input):
        feature_map = self.main_model(input)
        feature_map = feature_map.view(feature_map.shape[0],-1)
        output = self.classifies_layer(feature_map)
        return output

