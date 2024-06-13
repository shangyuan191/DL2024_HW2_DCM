import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.models import resnet34, ResNet34_Weights  # Import weights
import torch.nn.functional as F

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
    
# 多尺度卷积模块，垂直深度叠加
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleConv, self).__init__()
        # 定义三种不同卷积核大小的卷积层，垂直深度叠加
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=7, padding=3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 连续应用不同卷积核大小的卷积层
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        return out

# 自注意力机制模块
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height)
        key = self.key(x).view(batch_size, -1, width * height)
        value = self.value(x).view(batch_size, -1, width * height)

        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x

        return out

# RRDB模块
class RRDB(nn.Module):
    def __init__(self, in_channels):
        super(RRDB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out + x

# GCN模块
class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# 集成CNN
class EnsembleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(EnsembleCNN, self).__init__()
        self.rrdb = RRDB(input_channels)
        self.attention = SelfAttention(input_channels)
        self.gcn = GCNLayer(input_channels, 64)
        self.multi_scale_conv = MultiScaleConv(input_channels)

        self.fc1 = nn.Linear(64*224*224*4, 512)  # 根据拼接后的维度进行调整
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        rrdb_out = self.rrdb(x)
        attention_out = self.attention(x)
        gcn_out = self.gcn(x)
        multi_scale_out = self.multi_scale_conv(x)

        combined_out = torch.cat([rrdb_out, attention_out, gcn_out, multi_scale_out], dim=1)
        combined_out = combined_out.view(combined_out.size(0), -1)  # 展平张量
        x = self.relu(self.fc1(combined_out))
        x = self.fc2(x)
        return x

# 示例运行
if __name__ == "__main__":
    model = EnsembleCNN(input_channels=3, num_classes=50)
    print(model)
    # input_tensor = torch.randn(1, 3, 224, 224)  # 输入张量
    # output_tensor = model(input_tensor)  # 运行模型
    # print(output_tensor.shape)