import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet34, ResNet34_Weights  # Import weights

class DynamicInputCNN(nn.Module):
    def __init__(self,num_classes=50):
        super(DynamicInputCNN,self).__init__()
        self.model=models.resnet34(weights=ResNet34_Weights.DEFAULT)

        self.model.conv1=nn.Conv2d(1, 64,kernel_size=7,stride=2,padding=3,bias=False)

        num_ftrs=self.model.fc.in_features
        self.model.fc=nn.Linear(num_ftrs,num_classes)

        self.model = nn.Sequential(*[PoolChannelAttention(), 
                                     self.model])

    def forward(self,x):
        return self.model(x)
    

class PoolChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        pool = ['Avg', 'Max']

        in_channels = 1
        self.kernel_size = 3

        self.pool_types = [getattr(nn, 'Adaptive' + p + 'Pool3d')((1, None, None)) for p in pool]

        ff_act = nn.ReLU()

        _Sequential = (
                [nn.Conv2d(1, 16, 3),
                 ff_act,
                 nn.Conv2d(16, 1, 3)])

        self.mlp = nn.Sequential(*_Sequential)
    def forward(self, x):
        y = None
        for pool_type in self.pool_types:
            pool_x = pool_type(x)
            out = self.mlp(pool_x)

            if y is None:
                y = out
            else:
                y += out

        return y