import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.transforms import v2
import torch
import torch.nn.functional as F
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self,txt_file,transform=None,target_channel='RGB',channel_method=None,loader_name=None):
        self.image_labels=[]
        with open(txt_file,'r') as file:
            for line in file:
                path,label=line.strip().split()
                self.image_labels.append(("./dataset/"+path,int(label)))
        self.transform=transform
        self.target_channel=target_channel
        self.channel_method=channel_method
        self.loader_name=loader_name

    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self,idx):
        img_path,label=self.image_labels[idx]
        image=read_image(img_path)
        if self.channel_method=="DIY":
            image=self.process_channel_DIY(image,self.target_channel)
        elif self.channel_method=="sigmoid":
            image=self.process_channel_by_sigmoid(image,self.target_channel)
        elif self.channel_method=="my_cool_model":
            image=self.process_channel_DIY(image,self.target_channel)
        elif self.channel_method=="baseline_model":
            image=self.process_channel_DIY(image,"RGB")
            image=self.processing_channel_for_baseline_model(image,self.target_channel)
        image=self.transform(image)
        return image,label
    
    def process_channel_DIY(self,image,target_channel):
        channels,_,_=image.shape
        if channels==1:
            image=image.repeat(3,1,1)

        if target_channel=="RGB":
            pass
        elif target_channel=="RG":
            image=image[:2,:,:]
        elif target_channel=="GB":
            image=image[1:,:,:]
        elif target_channel=="R":
            if image.size(0)==3:
                image=image[0,:,:].unsqueeze(0)
        elif target_channel=="G":
            if image.size(0)==3:
                image=image[1,:,:].unsqueeze(0)
        elif target_channel=="B":
            if image.size(0)==3:
                image=image[2,:,:].unsqueeze(0)

        return image
    
    def processing_channel_for_baseline_model(self,image,target_channel):
        # Create a zeroed image with 3 channels
        processed_image = torch.zeros_like(image)
        
        # Map channels to indices
        channel_map = {'R': 0, 'G': 1, 'B': 2}
        
        # Copy the necessary channels to the processed_image
        for c in target_channel:
            processed_image[channel_map[c]] = image[channel_map[c]]
        
        return processed_image
    
    def process_channel_by_sigmoid(self,image,target_channel):
        channel_dic = {'R': 0, 'G': 1, 'B': 2}
        target_channel_indices = [channel_dic[ch] for ch in target_channel]
        num_channels = image.shape[0]

        if num_channels == 1:
            image = F.sigmoid(image)
            image = torch.cat([image] * len(target_channel), dim=0)
        elif num_channels == 3:
            image = image[target_channel_indices, :, :]
            image = F.sigmoid(image)

        return image
    


# Define a function to get normalization parameters based on target_channel
def get_normalize_params(target_channel):
    if target_channel == 'RGB':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif target_channel in ['RG', 'GB']:
        mean = [0.485, 0.456][:2]  # Using only two channels
        std = [0.229, 0.224][:2]   # Using only two channels
    elif target_channel in ['R', 'G', 'B']:
        mean = [0.485]  # Using only one channel
        std = [0.229]   # Using only one channel
    return mean, std



def build_train_loader(batch_size,num_workers,target_channel,channel_method):
    mean, std = get_normalize_params(target_channel)
    image_dtype = torch.uint8
    _transforms_normal = [v2.RandomResizedCrop(size=(224, 224), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
                v2.ToDtype(image_dtype, scale=True)]
    _transforms_G=[v2.RandomResizedCrop(size=(224, 224)),v2.ToDtype(torch.float32, scale=True),
                   v2.Normalize(mean=mean, std=std),
                   v2.ToDtype(image_dtype, scale=True)]
    _transforms = v2.Compose(_transforms_normal)
    if channel_method=="task2_resnet34_model":
        _transforms=v2.Compose(_transforms_G)
        print("use G preprocessing")

    train_dataset=ImageDataset(txt_file='./dataset/train.txt',transform=_transforms,target_channel=target_channel,channel_method=channel_method)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    return train_loader


def build_val_loader(batch_size,num_workers,target_channel,channel_method):
    mean, std = get_normalize_params(target_channel)
    image_dtype = torch.uint8
    _transforms_normal = [v2.RandomResizedCrop(size=(224, 224), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
                v2.ToDtype(image_dtype, scale=True)]
    _transforms_G=[v2.RandomResizedCrop(size=(224, 224)),v2.ToDtype(torch.float32, scale=True),
                   v2.Normalize(mean=mean, std=std),
                   v2.ToDtype(image_dtype, scale=True)]
    _transforms = v2.Compose(_transforms_normal)
    if channel_method=="task2_resnet34_model":
        _transforms=v2.Compose(_transforms_G)
        print("use G preprocessing")

    val_dataset=ImageDataset(txt_file='./dataset/val.txt',transform=_transforms,target_channel=target_channel,channel_method=channel_method)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    return val_loader


def build_test_loader(batch_size,num_workers,target_channel,channel_method):
    mean, std = get_normalize_params(target_channel)
    if channel_method=="baseline_model":
        mean, std = get_normalize_params("RGB")
    image_dtype = torch.uint8
    _transforms_normal = [v2.RandomResizedCrop(size=(224, 224), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
                v2.ToDtype(image_dtype, scale=True)]
    _transforms_G=[v2.RandomResizedCrop(size=(224, 224)),v2.ToDtype(torch.float32, scale=True),
                   v2.Normalize(mean=mean, std=std),
                   v2.ToDtype(image_dtype, scale=True)]
    _transforms = v2.Compose(_transforms_normal)
    if channel_method=="task2_resnet34_model":
        _transforms=v2.Compose(_transforms_G)
        print("use G preprocessing")

    test_dataset=ImageDataset(txt_file='./dataset/test.txt',transform=_transforms,target_channel=target_channel,channel_method=channel_method)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    return test_loader




