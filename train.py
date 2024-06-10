import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34
from tqdm import tqdm
from my_model import my_cool_model



def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss=0.0
    for images, labels in tqdm(train_loader,desc='train_loader'):
        images, labels = images.to(device).float(), labels.to(device)  # Ensure images are FloatTensor
        optimizer.zero_grad()
        with torch.autocast(device_type=device):
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()*images.size(0)
    epoch_loss=running_loss/len(train_loader.dataset)
    return epoch_loss
def validate_model(model, val_loader, criterion, device):
    model.eval()
    total, correct = 0, 0
    val_loss=0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader,desc='validate_loader'):
            images, labels = images.to(device).float(), labels.to(device)  # Ensure images are FloatTensor
            outputs = model(images)
            loss=criterion(outputs,labels)
            val_loss+=loss.item()*images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss/=total
    val_accuracy = correct / total
    return val_loss,val_accuracy

def build_model_and_optimizer(target_channel, num_classes, device, model_type="resnet34"):
    if model_type == "resnet34":
        if target_channel == "RGB":
            model = resnet34(weights='IMAGENET1K_V1')  # Use updated argument for weights
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model = resnet34(weights='IMAGENET1K_V1')
            model.conv1 = nn.Conv2d(len(target_channel), 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type=='my_cool_model':
        model=my_cool_model.DynamicInputCNN(num_classes)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    return model, criterion, optimizer