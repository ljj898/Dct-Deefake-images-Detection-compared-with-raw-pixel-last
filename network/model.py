import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResBlockDown, SelfAttention, ResBlock, ResBlockD, ResBlockUp
import math
import sys
import os
from tqdm import tqdm

#components
class Discriminator(nn.Module):
    def __init__(self, in_height):
        super(Discriminator, self).__init__()
        
        self.relu = nn.LeakyReLU(inplace=False)
        
        #in 6*256*256
        self.resDown1 = ResBlockDown(3, 64) #out 64*128*128
        self.resDown2 = ResBlockDown(64, 128) #out 128*64*64
        self.resDown3 = ResBlockDown(128, 256) #out 256*32*32
        self.self_att = SelfAttention(256) #out 256*32*32
        self.resDown4 = ResBlockDown(256, 512) #out 515*16*16
        self.resDown5 = ResBlockDown(512, 512) #out 512*8*8
        self.resDown6 = ResBlockDown(512, 512) #out 512*4*4
        self.sum_pooling = nn.AdaptiveMaxPool2d((1,1)) #out 512*1*1
        self.fc = nn.Sequential(nn.ReLU(True), nn.Linear(512, 5))

    def forward(self, x):
        out = self.resDown1(x) #out 64*128*128
        out = self.resDown2(out) #out 128*64*64
        out = self.resDown3(out) #out 256*32*32
        
        out = self.self_att(out) #out 256*32*32
        
        out = self.resDown4(out) #out 512*16*16
        out = self.resDown5(out) #out 512*8*8
        out = self.resDown6(out) #out 512*4*4
        
        out = self.sum_pooling(out) #out 512*1*1
        out = self.relu(out) #out 512*1*1
        out = out.view(-1,512) #out B*512
        out = self.fc(out)
        return out

class Generator(nn.Module):    
    def __init__(self, in_height=256, finetuning=False, e_finetuning=None):
        super(Generator, self).__init__()
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(inplace = False)
        
        #in 3*256*256
        #Down
        self.resDown1 = ResBlockDown(3, 64, conv_size=9, padding_size=4) #out 64*128*128
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        
        self.resDown2 = ResBlockDown(64, 128) #out 128*64*64
        self.in2 = nn.InstanceNorm2d(128, affine=True)
        
        self.resDown3 = ResBlockDown(128, 256) #out 256*32*32
        self.in3 = nn.InstanceNorm2d(256, affine=True)
        
        self.self_att_Down = SelfAttention(256) #out 256*32*32
        
        self.resDown4 = ResBlockDown(256, 512) #out 512*16*16
        self.in4 = nn.InstanceNorm2d(512, affine=True)
        
        #Res
        #in 512*16*16
        self.res1 = ResBlock(512)
        self.res2 = ResBlock(512)
        self.res3 = ResBlock(512)
        self.res4 = ResBlock(512)
        self.res5 = ResBlock(512)
        #out 512*16*16
        
        #Up
        #in 512*16*16
        self.resUp1 = ResBlockUp(512, 256) #out 256*32*32
        self.resUp2 = ResBlockUp(256, 128) #out 128*64*64
        
        self.self_att_Up = SelfAttention(128) #out 128*64*64

        self.resUp3 = ResBlockUp(128, 64) #out 64*128*128
        self.resUp4 = ResBlockUp(64, 32, out_size=(in_height, in_height), scale=None, conv_size=3, padding_size=1) #out 3*256*256
        self.conv2d = nn.Conv2d(32, 3, 3, padding = 1)
        
            
    def forward(self, x):

        #in 3*256*256
        #Encoding
        out = self.resDown1(x)
        out = self.in1(out)
        
        out = self.resDown2(out)
        out = self.in2(out)
        
        out = self.resDown3(out)
        out = self.in3(out)
        
        out = self.self_att_Down(out)
        
        out = self.resDown4(out)
        out = self.in4(out)
        
        
        #Residual
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        
        
        #Decoding
        out = self.resUp1(out)
        
        out = self.resUp2(out)
        
        out = self.self_att_Up(out)

        out = self.resUp3(out)
        
        out = self.resUp4(out)
        
        out = self.relu(out)
        
        out = self.conv2d(out)
        
        out = self.sigmoid(out)
        
        #out = out*255
        
        #out 3*224*224
        return out

# class W_i_class(nn.Module):
#     def __init__(self):
#         super(W_i_class, self).__init__()
#         self.W_i = nn.Parameter(torch.randn(512,2))
    
#     def forward(self):
#         return self.W_i


class Cropped_VGG19(nn.Module):
    def __init__(self):
        super(Cropped_VGG19, self).__init__()
        
        self.conv1_1 = nn.Conv2d(3,64,3)
        self.conv1_2 = nn.Conv2d(64,64,3)
        self.conv2_1 = nn.Conv2d(64,128,3)
        self.conv2_2 = nn.Conv2d(128,128,3)
        self.conv3_1 = nn.Conv2d(128,256,3)
        self.conv3_2 = nn.Conv2d(256,256,3)
        self.conv3_3 = nn.Conv2d(256,256,3)
        self.conv4_1 = nn.Conv2d(256,512,3)
        self.conv4_2 = nn.Conv2d(512,512,3)
        self.conv4_3 = nn.Conv2d(512,512,3)
        self.conv5_1 = nn.Conv2d(512,512,3)
        #self.conv5_2 = nn.Conv2d(512,512,3)
        #self.conv5_3 = nn.Conv2d(512,512,3)
        
    def forward(self, x):
        conv1_1_pad     = F.pad(x, (1, 1, 1, 1))
        conv1_1         = self.conv1_1(conv1_1_pad)
        relu1_1         = F.relu(conv1_1)
        conv1_2_pad     = F.pad(relu1_1, (1, 1, 1, 1))
        conv1_2         = self.conv1_2(conv1_2_pad)
        relu1_2         = F.relu(conv1_2)
        pool1_pad       = F.pad(relu1_2, (0, 1, 0, 1), value=float('-inf'))
        pool1           = F.max_pool2d(pool1_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv2_1_pad     = F.pad(pool1, (1, 1, 1, 1))
        conv2_1         = self.conv2_1(conv2_1_pad)
        relu2_1         = F.relu(conv2_1)
        conv2_2_pad     = F.pad(relu2_1, (1, 1, 1, 1))
        conv2_2         = self.conv2_2(conv2_2_pad)
        relu2_2         = F.relu(conv2_2)
        pool2_pad       = F.pad(relu2_2, (0, 1, 0, 1), value=float('-inf'))
        pool2           = F.max_pool2d(pool2_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv3_1_pad     = F.pad(pool2, (1, 1, 1, 1))
        conv3_1         = self.conv3_1(conv3_1_pad)
        relu3_1         = F.relu(conv3_1)
        conv3_2_pad     = F.pad(relu3_1, (1, 1, 1, 1))
        conv3_2         = self.conv3_2(conv3_2_pad)
        relu3_2         = F.relu(conv3_2)
        conv3_3_pad     = F.pad(relu3_2, (1, 1, 1, 1))
        conv3_3         = self.conv3_3(conv3_3_pad)
        relu3_3         = F.relu(conv3_3)
        pool3_pad       = F.pad(relu3_3, (0, 1, 0, 1), value=float('-inf'))
        pool3           = F.max_pool2d(pool3_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv4_1_pad     = F.pad(pool3, (1, 1, 1, 1))
        conv4_1         = self.conv4_1(conv4_1_pad)
        relu4_1         = F.relu(conv4_1)
        conv4_2_pad     = F.pad(relu4_1, (1, 1, 1, 1))
        conv4_2         = self.conv4_2(conv4_2_pad)
        relu4_2         = F.relu(conv4_2)
        conv4_3_pad     = F.pad(relu4_2, (1, 1, 1, 1))
        conv4_3         = self.conv4_3(conv4_3_pad)
        relu4_3         = F.relu(conv4_3)
        pool4_pad       = F.pad(relu4_3, (0, 1, 0, 1), value=float('-inf'))
        pool4           = F.max_pool2d(pool4_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv5_1_pad     = F.pad(pool4, (1, 1, 1, 1))
        conv5_1         = self.conv5_1(conv5_1_pad)
        relu5_1         = F.relu(conv5_1)
        
        return [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1]
    
