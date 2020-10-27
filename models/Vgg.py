import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from utils.srm import SRMLayer 
 
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'myvgg': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
}
 
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 2)
 
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
 
    def _make_layers(self, cfg):
        layers = []
        in_channels = 6
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                
        layers += [nn.Conv2d(512, 512, kernel_size=7, padding=0)]
        return nn.Sequential(*layers)

class Vgg16(nn.Module):
    def __init__(self, pretrained, use_srm):
        super(Vgg16, self).__init__()
        self.use_srm = use_srm
        if self.use_srm:
            self.srm = SRMLayer()
        self.model = models.vgg16(pretrained)
        self.exfc = nn.Sequential(nn.ReLU(True), nn.Linear(1000, 2))

    def forward(self, x):
        if self.use_srm:
            x = self.srm(x)
        x = self.model(x)
        return self.exfc(x)

class Vgg2x(nn.Module):
    def __init__(self, pretrained):
        super(Vgg2x, self).__init__()
        self.model1 = models.vgg16(pretrained)
        self.model2 = models.vgg16(pretrained)
        self.exfc = nn.Sequential(nn.ReLU(True), nn.Linear(2000, 3))

    def forward(self, x):
        a = self.model1(x[:, [0]].squeeze(1))
        b = self.model2(x[:, [1]].squeeze(1))
        a = torch.cat((a, b), 1)
        return self.exfc(a)


def main():

    net = VGG('VGG16')
    x = torch.randn(1, 3, 224, 224)
    print(net(x).size())

if __name__ == '__main__':
    main()

        
        