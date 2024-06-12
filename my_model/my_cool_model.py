import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.models import resnet34, ResNet34_Weights  # Import weights
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.fc1 = nn.Linear(128, 512)  # Assuming input image size is 224*224
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool2(x).flatten(1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DynamicInputCNN(nn.Module):
    def __init__(self,target_channel,num_classes=50):
        super(DynamicInputCNN,self).__init__()
        self.model=SimpleCNN(1,num_classes)

        self.model.conv1=nn.Conv2d(1, 32,kernel_size=3,padding=1)

        num_ftrs=self.model.fc2.in_features
        self.model.fc2=nn.Linear(num_ftrs,num_classes)

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
    

class CustomVGG16(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CustomVGG16, self).__init__()
        self.model = models.vgg16(pretrained=False)
        self.model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)