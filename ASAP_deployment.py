"""
Code for the article
ASAP: Automated Style-Aware Similarity Measurement for Selection of Annotated Pre-training Datasets in 2D Biomedical Imaging
Miguel Molina-Moreno, Marcel P. Schilling, Markus Reischl, Ralf Mikut
IEEE Access

Code based on:
    
https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py
https://github.com/Horizon2333/imagenet-autoencoder/blob/main/models/resnet.py

@author: Miguel Molina Moreno (miguel.molinamoreno@yale.edu)
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as FT
import glob
from PIL import Image
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import pandas as pd
import sys

# Style loss computation and Gram matrix functions
def gram_matrix(input):
    a, b, c, d = input.size()  
    # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a, b, c * d)  # resise F_XL into \hat F_XL

    G = torch.zeros((features.size(0),features.size(1),features.size(1)),dtype=features.dtype,device=features.device)
    for i in range(features.size(0)):
        G[i,:,:] = torch.mm(features[i,:,:], features[i,:,:].t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(b)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# ResNet AutoEncoder
def get_configs(arch='resnet50'):

    # True or False means wether to use BottleNeck

    if arch == 'resnet18':
        return [2, 2, 2, 2], False
    elif arch == 'resnet34':
        return [3, 4, 6, 3], False
    elif arch == 'resnet50':
        return [3, 4, 6, 3], True
    elif arch == 'resnet101':
        return [3, 4, 23, 3], True
    elif arch == 'resnet152':
        return [3, 8, 36, 3], True
    else:
        raise ValueError("Undefined model")

class ResNetAutoEncoder(nn.Module):

    def __init__(self, configs, bottleneck):

        super(ResNetAutoEncoder, self).__init__()

        self.encoder = ResNetEncoder(configs=configs,       bottleneck=bottleneck)
        self.decoder = ResNetDecoder(configs=configs[::-1], bottleneck=bottleneck)
    
    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

class ResNet(nn.Module):

    def __init__(self, configs, bottleneck=False, num_classes=1000):
        super(ResNet, self).__init__()

        self.encoder = ResNetEncoder(configs, bottleneck)

        self.avpool = nn.AdaptiveAvgPool2d((1,1))

        if bottleneck:
            self.fc = nn.Linear(in_features=2048, out_features=num_classes)
        else:
            self.fc = nn.Linear(in_features=512, out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        x = self.encoder(x)

        x = self.avpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


class ResNetEncoder(nn.Module):

    def __init__(self, configs, bottleneck=False):
        super(ResNetEncoder, self).__init__()

        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configued")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        if bottleneck:

            self.conv2 = EncoderBottleneckBlock(in_channels=64,   hidden_channels=64,  up_channels=256,  layers=configs[0], downsample_method="pool")
            self.conv3 = EncoderBottleneckBlock(in_channels=256,  hidden_channels=128, up_channels=512,  layers=configs[1], downsample_method="conv")
            self.conv4 = EncoderBottleneckBlock(in_channels=512,  hidden_channels=256, up_channels=1024, layers=configs[2], downsample_method="conv")
            self.conv5 = EncoderBottleneckBlock(in_channels=1024, hidden_channels=512, up_channels=2048, layers=configs[3], downsample_method="conv")

        else:

            self.conv2 = EncoderResidualBlock(in_channels=64,  hidden_channels=64,  layers=configs[0], downsample_method="pool")
            self.conv3 = EncoderResidualBlock(in_channels=64,  hidden_channels=128, layers=configs[1], downsample_method="conv")
            self.conv4 = EncoderResidualBlock(in_channels=128, hidden_channels=256, layers=configs[2], downsample_method="conv")
            self.conv5 = EncoderResidualBlock(in_channels=256, hidden_channels=512, layers=configs[3], downsample_method="conv")

    def forward(self, x):

        x = self.conv1(x)
        gram1=gram_matrix(x)
        x = self.conv2(x)
        gram2=gram_matrix(x)
        x = self.conv3(x)
        gram3=gram_matrix(x)
        x = self.conv4(x)
        gram4=gram_matrix(x)
        x = self.conv5(x)

        return x, gram1, gram2, gram3, gram4

class ResNetDecoder(nn.Module):

    def __init__(self, configs, bottleneck=False):
        super(ResNetDecoder, self).__init__()

        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configued")

        if bottleneck:

            self.conv1 = DecoderBottleneckBlock(in_channels=2048, hidden_channels=512, down_channels=1024, layers=configs[0])
            self.conv2 = DecoderBottleneckBlock(in_channels=1024, hidden_channels=256, down_channels=512,  layers=configs[1])
            self.conv3 = DecoderBottleneckBlock(in_channels=512,  hidden_channels=128, down_channels=256,  layers=configs[2])
            self.conv4 = DecoderBottleneckBlock(in_channels=256,  hidden_channels=64,  down_channels=64,   layers=configs[3])


        else:

            self.conv1 = DecoderResidualBlock(hidden_channels=512, output_channels=256, layers=configs[0])
            self.conv2 = DecoderResidualBlock(hidden_channels=256, output_channels=128, layers=configs[1])
            self.conv3 = DecoderResidualBlock(hidden_channels=128, output_channels=64,  layers=configs[2])
            self.conv4 = DecoderResidualBlock(hidden_channels=64,  output_channels=64,  layers=configs[3])

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
        )

        self.gate = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        gram4=gram_matrix(x)
        x = self.conv2(x)
        gram3=gram_matrix(x)
        x = self.conv3(x)
        gram2=gram_matrix(x)
        x = self.conv4(x)
        gram1=gram_matrix(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x, gram1, gram2, gram3, gram4

class EncoderResidualBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, layers, downsample_method="conv"):
        super(EncoderResidualBlock, self).__init__()

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, downsample=True)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % i, layer)
        
        elif downsample_method == "pool":

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, downsample=False)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % (i+1), layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class EncoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, up_channels, layers, downsample_method="conv"):
        super(EncoderBottleneckBlock, self).__init__()

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=True)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % i, layer)
        
        elif downsample_method == "pool":

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % (i+1), layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x


class DecoderResidualBlock(nn.Module):

    def __init__(self, hidden_channels, output_channels, layers):
        super(DecoderResidualBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=output_channels, upsample=True)
            else:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=hidden_channels, upsample=False)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class DecoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, layers):
        super(DecoderBottleneckBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=down_channels, upsample=True)
            else:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=in_channels, upsample=False)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x


class EncoderResidualLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, downsample):
        super(EncoderResidualLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        x = self.relu(x)

        return x

class EncoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, up_channels, downsample):
        super(EncoderBottleneckLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.weight_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=up_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        elif (in_channels != up_channels):
            self.downsample = None
            self.up_scale = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        else:
            self.downsample = None
            self.up_scale = None

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        elif self.up_scale is not None:
            identity = self.up_scale(identity)

        x = x + identity

        x = self.relu(x)

        return x

class DecoderResidualLayer(nn.Module):

    def __init__(self, hidden_channels, output_channels, upsample):
        super(DecoderResidualLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if upsample:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)                
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=1, stride=2, output_padding=1, bias=False)   
            )
        else:
            self.upsample = None
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity

        return x

class DecoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, upsample):
        super(DecoderBottleneckLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.weight_layer2 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if upsample:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
            )
        else:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
            )
        elif (in_channels != down_channels):
            self.upsample = None
            self.down_scale = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )
        else:
            self.upsample = None
            self.down_scale = None
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.upsample is not None:
            identity = self.upsample(identity)
        elif self.down_scale is not None:
            identity = self.down_scale(identity)

        x = x + identity

        return x


class AE_ResNet(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        configs, bottleneck = get_configs("resnet18")
        self.encoder = ResNetEncoder(configs, bottleneck)
        self.decoder = ResNetDecoder(configs[::-1], bottleneck)
        self.linear1_enc = nn.Linear(512, 256)
        self.linear2_enc = nn.Linear(256, z_dim)
        self.linear1_dec = nn.Linear(z_dim, 256)
        self.linear2_dec = nn.Linear(256, 512)
        
    def forward(self, x):
        z, gram1_enc, gram2_enc, gram3_enc, gram4_enc = self.encoder(x)

        z = F.adaptive_avg_pool2d(z, 1)
        z = z.view(z.size(0), -1)
        z = torch.relu(self.linear1_enc(z))
        z = self.linear2_enc(z)


        x = torch.relu(self.linear1_dec(z))
        x = self.linear2_dec(x)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=2)

        x, gram1_dec, gram2_dec, gram3_dec, gram4_dec = self.decoder(x)

        return x, z, gram1_enc, gram2_enc, gram3_enc, gram4_enc, gram1_dec, gram2_dec, gram3_dec, gram4_dec
    
# Dataset handler and data augmentation  
def custom_data_augmentation(img,size):
    # Random gamma adjustment
    gamma = 0.1*np.random.randn()+1
    if (gamma < 0.5):
        gamma = 0.5
    elif (gamma > 2):
        gamma = 2
    img = FT.adjust_gamma(img, gamma, 1)
    # Random brightness adjustment
    brightness_factor = 0.1*np.random.randn()+1
    img = FT.adjust_brightness(img, brightness_factor)
    # Random contrast adjustment
    contrast_factor = 0.1*np.random.randn()+1
    img = FT.adjust_contrast(img, contrast_factor)
    # Random scale
    angle = 0
    shear = 0
    scale = 1+0.5*(np.random.random()-0.5)
    translate = [0,0]
    img = FT.affine(img, angle, translate, scale, shear)#, resample=0, fillcolor=None
    # Random flip
    if (np.random.rand()>0.5):
        img = FT.hflip(img)
    if (np.random.rand()>0.5):
        img = FT.vflip(img)
    if (np.random.rand()>0.5):
        # Before cropping, size of 2xsize. This way each crop has around 25% of the image
        img = FT.resize(img, (2*size[0],2*size[1]))
        x_min=np.random.randint(img.size[0]-size[1])
        y_min=np.random.randint(img.size[1]-size[0])
        img = FT.crop(img, y_min, x_min, size[0],size[1])
    else:
        img = FT.resize(img, (size[0],size[1]))

    return img
    
class CustomDataset(Dataset):
    def __init__(self, db_root,extension,size,augm=False):
        self.imgs = glob.glob(os.path.join(db_root,'**','images','*'+extension))
        self.size = size
        self.augm = augm
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        if (not img.mode=="RGB"):
            img=Image.fromarray(np.repeat(np.array(img)[:,:,np.newaxis],3,axis=2))
        if (self.augm):
            img = custom_data_augmentation(img, self.size)
        else:
            img = FT.resize(img, (self.size[0],self.size[1]))
        img = torch.as_tensor(np.array(img)/255.0).permute(2,0,1).type(torch.FloatTensor)
        dataset_id = int(self.imgs[idx].split(os.path.sep)[-3])
        return img, dataset_id
    
class CustomDatasetTriplet(Dataset):
    def __init__(self, db_root,extension,size,augm=False):
        self.imgs = glob.glob(os.path.join(db_root,'**','images','*'+extension))
        self.size = size
        self.augm = augm
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        if (not img.mode=="RGB"):
            img=Image.fromarray(np.repeat(np.array(img)[:,:,np.newaxis],3,axis=2))
        if (self.augm):
            img = custom_data_augmentation(img, self.size)
        else:
            img = FT.resize(img, (self.size[0],self.size[1]))
        img = torch.as_tensor(np.array(img)/255.0).permute(2,0,1).type(torch.FloatTensor)
        dataset_id = int(self.imgs[idx].split(os.path.sep)[-3])
        # Get a positive
        imgs_pos=[x for x in self.imgs if (os.path.sep+self.imgs[idx].split(os.path.sep)[-3]+os.path.sep) in x]
        img_pos_path=imgs_pos[np.random.randint(len(imgs_pos))]
        img_pos = Image.open(img_pos_path).convert("RGB")
        if (not img_pos.mode=="RGB"):
            img_pos=Image.fromarray(np.repeat(np.array(img_pos)[:,:,np.newaxis],3,axis=2))
        if (self.augm):
            img_pos = custom_data_augmentation(img_pos, self.size)
        else:
            img_pos = FT.resize(img_pos, (self.size[0],self.size[1]))
        img_pos = torch.as_tensor(np.array(img_pos)/255.0).permute(2,0,1).type(torch.FloatTensor)
        # Get a negative
        imgs_neg=[x for x in self.imgs if (os.path.sep+self.imgs[idx].split(os.path.sep)[-3]+os.path.sep) not in x]
        img_neg_path=imgs_neg[np.random.randint(len(imgs_neg))]
        img_neg = Image.open(img_neg_path).convert("RGB")
        if (not img_neg.mode=="RGB"):
            img_neg=Image.fromarray(np.repeat(np.array(img_neg)[:,:,np.newaxis],3,axis=2))
        if (self.augm):
            img_neg = custom_data_augmentation(img_neg, self.size)
        else:
            img_neg = FT.resize(img_neg, (self.size[0],self.size[1]))
        img_neg = torch.as_tensor(np.array(img_neg)/255.0).permute(2,0,1).type(torch.FloatTensor)

        return img, img_pos, img_neg, dataset_id
    
# Function that calculate the mean, minimum and maximum distances and the EQC between a pair of datasets
def calculate_distances(x,y):
    distances=distance_matrix(x,x,p=2)
    
    # Mean and minimum distances
    ids=np.unique(y)
    mean_distances=np.zeros((ids.shape[0],ids.shape[0]),dtype='float')
    min_distances=np.zeros((ids.shape[0],ids.shape[0]),dtype='float')
    max_distances=np.zeros((ids.shape[0],ids.shape[0]),dtype='float')
    EQC=np.zeros((ids.shape[0],ids.shape[0]),dtype='float')
    
    
    for i in range(ids.shape[0]):
        for j in range(ids.shape[0]):
            submatrix=distances[np.ix_(np.where(y==ids[i])[0],np.where(y==ids[j])[0])]
            mean_distances[i,j]=np.mean(submatrix)
            min_distances[i,j]=np.min(submatrix)
            max_distances[i,j]=np.max(submatrix)
            if (i==j):
                EQC[i,j]=-float('inf')
            else:
                submatrixI=x[np.ix_(np.where(y==ids[i])[0],np.arange(x.shape[1]))]
                submatrixJ=x[np.ix_(np.where(y==ids[j])[0],np.arange(x.shape[1]))]
                distancesI=distance_matrix(submatrixI,submatrixI,p=2)
                distancesJ=distance_matrix(submatrixJ,submatrixJ,p=2)
                EQC[i,j]=(np.mean(distancesI)+np.mean(distancesJ))/np.mean(submatrix)
    return mean_distances, min_distances, max_distances, EQC
    
if __name__ == '__main__':

    # Parameters
    eConfig = {'alpha': 0.5,
            'beta':0.6,
            'weight1':0.2,
            'weight2':0.2,
            'weight3':0.3,
            'weight4':0.3,
            'z_dim':2}
        
    args = sys.argv[1::]
    for i in range(0,len(args),2):
        key = args[i]
        val = args[i+1]
        eConfig[key] = type(eConfig[key])(val)
        print (str(eConfig[key]))
          
    print('args')
    print(sys.argv)
    print('eConfig')
    print(eConfig)
    style_weights=[eConfig['weight1'],eConfig['weight2'],eConfig['weight3'],eConfig['weight4']] # gram1, gram2, gram3, gram4

        
    # Data and results options
    DATASET_FOLDER = 'dataset_full'
    RESULT_FOLDER = 'results_z_'+str(eConfig['z_dim'])+'_deployment'
    extension = '.png'
    input_size = [64,64,3] # [Height, Width, Channels]
    z_dim = eConfig['z_dim']
    n_train_datasets=21
    train_dataset_names=['HAM10000'+' ('+r'$D^{a}_1$'+')','MoNuSeg'+' ('+r'$D^{a}_2$'+')','PanNuKe'+' ('+r'$D^{a}_3$'+')','Fluo-N2DL-HeLa'+' ('+r'$D^{a}_4$'+')','Cellpose'+' ('+r'$D^{a}_5$'+')','PhC-C2DH-U373'+' ('+r'$D^{a}_6$'+')','URICADS'+' ('+r'$D^{a}_7$'+')','heartSeg'+' ('+r'$D^{a}_8$'+')','DENTAL'+' ('+r'$D^{a}_9$'+')','ISIC2017'+' ('+r'$D^{na}_1$'+')','CryoNuSeg'+' ('+r'$D^{na}_2$'+')','Fluo-N2DH-GOWT1'+' ('+r'$D^{na}_3$'+')','PhC-C2DL-PSC'+' ('+r'$D^{na}_4$'+')','LUMINOUS'+' ('+r'$D^{na}_5$'+')','Fluo-C2DL-Huh7'+' ('+r'$D^{na}_6$'+')','DIC-C2DH-HeLa'+' ('+r'$D^{na}_7$'+')','LIVECell'+' ('+r'$D^{na}_8$'+')','CVC-ClinicDB'+' ('+r'$D^{na}_9$'+')','USNerve'+' ('+r'$D^{na}_{10}$'+')','RIGA'+' ('+r'$D^{na}_{11}$'+')','LUNG'+' ('+r'$D^{na}_{12}$'+')']
    dataset_names=train_dataset_names.copy()
    if not os.path.exists(RESULT_FOLDER):
        os.mkdir(RESULT_FOLDER)
    # Load the segmentation results
    results_test=pd.read_csv('results_full.csv',decimal=',',header=None)
    results_test=results_test.to_numpy().astype('float32')
     
    # Plot options
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.set_style("whitegrid")    
    POINT_SIZE=60
    ALPHA=1
    palette = sns.color_palette("bright", 9)
    palette_test = sns.color_palette(np.concatenate((palette,sns.color_palette("pastel", 9),sns.color_palette("dark", 3)),axis=0))
    
    # Training options
    batch_size = 8
    lr = 0.001
    momentum = 0.9
    weight_decay = 5e-4
    step_size = 10
    gamma=0.5
    n_epochs = 100
    criterion=torch.nn.MSELoss()
    
    # Training AE with triplet loss
    epoch_loss=np.zeros((n_epochs,),dtype='float')
    best_loss=float('inf')
    best_model=[]
    triplet=torch.nn.TripletMarginLoss()
    
    # Load the pretrained model
    model = AE_ResNet(z_dim)
    pretrained = torch.load('caltech256-resnet18.pth')
    print('Pretrained model loaded')
    model_dict = model.state_dict()
    model_dict.update(pretrained['state_dict'])
    print('Model_dict_updated')
    model_dict=dict((key.split('module.')[-1],value) for (key,value) in zip(model_dict.keys(),model_dict.values()))
    model.load_state_dict(model_dict)
    print('Model dict loaded')

    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Send the model to GPU if available
    model = model.to(device)
    
    # Optimizer
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=momentum, weight_decay=weight_decay)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=step_size,
                                                   gamma=gamma)
    
    # Dataloaders
    print("Initializing Datasets and Dataloaders...")
    
    # use our dataset and defined transformations
    dataset_train = CustomDatasetTriplet(DATASET_FOLDER,extension,input_size,augm=True)
    
    # define training data loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=None)
    
    # Training loop
    model.train()
    
    if (not os.path.exists(os.path.join(RESULT_FOLDER, 'triplet_style_model_best.pth.tar'))):
        for epoch in range(1, n_epochs+1):
            # monitor training loss
            train_loss = 0.0
        
            #Training
            for data, data_pos, data_neg, dset_id in data_loader_train:
                images, images_pos, images_neg = data, data_pos, data_neg
                images, images_pos, images_neg = images.to(device), images_pos.to(device), images_neg.to(device)
                optimizer.zero_grad()
                outputD, outputE, gram1_enc, gram2_enc, gram3_enc, gram4_enc, gram1_dec, gram2_dec, gram3_dec, gram4_dec = model(images)
                _, output_posE, gram1_enc_pos, gram2_enc_pos, gram3_enc_pos, gram4_enc_pos, gram1_dec_pos, gram2_dec_pos, gram3_dec_pos, gram4_dec_pos = model(images_pos)
                _, output_negE, gram1_enc_neg, gram2_enc_neg, gram3_enc_neg, gram4_enc_neg, gram1_dec_neg, gram2_dec_neg, gram3_dec_neg, gram4_dec_neg = model(images_neg)
                loss_rec = criterion(outputD, images)
                style_loss = style_weights[0]*criterion(gram1_enc, gram1_dec)+style_weights[1]*criterion(gram2_enc, gram2_dec)+style_weights[2]*criterion(gram3_enc, gram3_dec)+style_weights[3]*criterion(gram4_enc, gram4_dec)
                loss1 = eConfig['beta']*loss_rec + (1-eConfig['beta'])*style_loss
                loss_triplet = triplet(outputE, output_posE, output_negE)
                style_loss_triplet=style_weights[0]*triplet(gram1_enc, gram1_enc_pos, gram1_enc_neg)+style_weights[1]*triplet(gram2_enc, gram2_enc_pos, gram2_enc_neg)+style_weights[2]*triplet(gram3_enc, gram3_enc_pos, gram3_enc_neg)+style_weights[3]*triplet(gram4_enc, gram4_enc_pos, gram4_enc_neg)
                loss2 = eConfig['beta']*loss_triplet + (1-eConfig['beta'])*style_loss_triplet
                loss = eConfig['alpha']*loss1 + (1-eConfig['alpha'])*loss2
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*images.size(0)
            
            lr_scheduler.step()
            train_loss = train_loss/len(data_loader_train)
            if (loss.item()<best_loss):
                best_loss=loss.item()
                best_model=model.state_dict().copy()
            epoch_loss[epoch-1]=train_loss
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
            
        torch.save({
            'state_dict': best_model,
            'min_loss': best_loss,
        }, os.path.join(RESULT_FOLDER, 'triplet_style_model_best.pth.tar'))
        
        fig1 = plt.gcf()
        plt.plot(epoch_loss)
        plt.ylabel('Train loss')
        fig1.savefig(os.path.join(RESULT_FOLDER,'triplet_style_train_loss.png'))
        plt.close('all') 
    else:
        weights=torch.load(os.path.join(RESULT_FOLDER, 'triplet_style_model_best.pth.tar'))['state_dict']
        model.load_state_dict(weights)
    
    # Obtain the embedding
    dataset_train = CustomDatasetTriplet(DATASET_FOLDER,extension,input_size,augm=False)
        
    # define data_loader with batch size 1
    data_loader_embedding = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=None)
    
    model.eval()
    outputs=np.zeros((len(data_loader_embedding),z_dim),dtype='float')
    dataset_ids=np.zeros((len(data_loader_embedding),),dtype='float')
    
    # Test the train embedding
    for i, (data, data_pos, data_neg,dset_id) in enumerate(data_loader_embedding):
        images, images_pos, images_neg = data, data_pos, data_neg
        images, images_pos, images_neg = images.to(device), images_pos.to(device), images_neg.to(device)
        _, outputs_aux, _, _, _, _, _, _, _, _  = model(images)
        outputs[i,:]=outputs_aux.detach().cpu().numpy()
        dataset_ids[i]=dset_id.cpu().numpy()
    # Graphical representation
    if (z_dim==2):
        # Graphical representation
        sns_plot = sns.scatterplot(x=outputs[:, 0], y=outputs[:, 1], hue=dataset_ids.ravel(), s=POINT_SIZE,
                                        legend='full', alpha=ALPHA, palette=palette_test)
    else:
        print('Implement your favourite dimensionality reduction algorithm: TSNE, UMAP, ...')

    plt.xlim([-6,10.5])
    sns_plot.set_xlabel(r'$z_1$',fontsize='18')
    sns_plot.set_ylabel(r'$z_2$',fontsize='18')
    new_labels = []
    for i in range(0, n_train_datasets):
        # replace labels
        new_labels.append(train_dataset_names[i])
    for t, l in zip(sns_plot.get_legend().texts, new_labels):  t.set_text(l)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.setp(sns_plot.get_legend().get_texts(), fontsize='14')
    plt.tight_layout()
    sns_plot.figure.savefig(os.path.join(RESULT_FOLDER, 'triplet_style_embedding.png'),dpi=600)
    plt.clf()

    
    # Calculate the EQC
    mean_matrix, min_matrix, max_matrix, EQC = calculate_distances(outputs,dataset_ids.ravel())
    EQC_full=np.mean(np.max(EQC*np.triu(np.ones((n_train_datasets,n_train_datasets),dtype='float'),k=0),axis=0))
    print('The quality criterion for the AE embedding is '+str(EQC_full))
