import torch
import torch.nn as nn

class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel, conv_size=3, padding_size=1):
        super(ResBlockDown, self).__init__()
        
        self.relu = nn.LeakyReLU()
        self.relu_inplace = nn.LeakyReLU(inplace = False)
        self.avg_pool2d = nn.AvgPool2d(2)
        
        #left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1,))
        
        #right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding = padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding = padding_size))

    def forward(self, x):
        res = x
        
        #left
        out_res = self.conv_l1(res)
        out_res = self.avg_pool2d(out_res)
        
        #right
        out = self.relu(x)
        out = self.conv_r1(out)
        out = self.relu_inplace(out)
        out = self.conv_r2(out)
        out = self.avg_pool2d(out)
        
        #merge
        out = out_res + out
        
        return out

class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()
        
        #conv f
        self.conv_f = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel//8, 1))
        #conv_g
        self.conv_g = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel//8, 1))
        #conv_h
        self.conv_h = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))
        
        self.softmax = nn.Softmax(-2) #sum in column j = 1
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        f_projection = self.conv_f(x) #BxC'xHxW, C'=C//8
        g_projection = self.conv_g(x) #BxC'xHxW
        h_projection = self.conv_h(x) #BxCxHxW
        
        f_projection = torch.transpose(f_projection.view(B,-1,H*W), 1, 2) #BxNxC', N=H*W
        g_projection = g_projection.view(B,-1,H*W) #BxC'xN
        h_projection = h_projection.view(B,-1,H*W) #BxCxN
        
        attention_map = torch.bmm(f_projection, g_projection) #BxNxN
        attention_map = self.softmax(attention_map) #sum_i_N (A i,j) = 1
        
        #sum_i_N (A i,j) = 1 hence oj = (HxAj) is a weighted sum of input columns
        out = torch.bmm(h_projection, attention_map) #BxCxN
        out = out.view(B,C,H,W)
        
        out = self.gamma*out + x
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResBlock, self).__init__()
        
        #using no ReLU method
        
        #general
        self.relu = nn.LeakyReLU(inplace = False)
        
        #left
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding = 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding = 1))
        
    def forward(self, x):
        C = psi_slice.shape[1]
        
        res = x
        
        out = self.relu(out)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = out + res
        
        return out
        
class ResBlockD(nn.Module):
    def __init__(self, in_channel):
        super(ResBlockD, self).__init__()
        
        #using no ReLU method
        
        #general
        self.relu = nn.LeakyReLU(inplace = False)
        
        #left
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding = 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding = 1))
        
    def forward(self, x):
        res = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = out + res
        
        return out

class ResBlockUp(nn.Module):
    def __init__(self, in_channel, out_channel, out_size=None, scale = 2, conv_size=3, padding_size = 1, is_bilinear = True):
        super(ResBlockUp, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        if is_bilinear:
            self.upsample = nn.Upsample(size = out_size, scale_factor=scale, mode='bilinear')
        else:
            self.upsample = nn.Upsample(size = out_size, scale_factor=scale)
        self.relu = nn.LeakyReLU(inplace = False)
        
        #left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1))
        
        #right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding = padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding = padding_size))
    
    def forward(self,x):
    
        res = x
        
        #left
        out_res = self.upsample(res)
        out_res = self.conv_l1(out_res)
        
        #right
        out = self.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = self.relu(out)
        out = self.conv_r2(out)
        
        out = out + out_res
        
        return out
