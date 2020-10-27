import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, frame_shape, num_of_layers=17):
        super(Generator, self).__init__()
        channels = 3
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        out = self.main(x)
        return out
    
        
class Discriminator(nn.Module):
    def __init__(self, frame_shape=256, conv_dim=64, c_dim=5, repeat_num=6, net='classfier'):
        super(Discriminator, self).__init__()
        self.net = net
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))

        kernel_size = int(frame_shape / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        if net == 'discriminator':
            self.last_conv = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        elif net == 'classfier':
            self.last_conv = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        out = self.main(x)
        out = self.last_conv(out)
        if self.net == 'classfier':
            out = out.view(out.shape[0], out.shape[1])
        return out


if __name__ == '__main__':
    from torchsummary import summary
    G = Generator(256)
    D = Discriminator(256)
    print(G)
    print(D)
    summary(G, input_size=(3,256,256), batch_size=-1, device='cpu')
    summary(D, input_size=(3,256,256), batch_size=-1, device='cpu')