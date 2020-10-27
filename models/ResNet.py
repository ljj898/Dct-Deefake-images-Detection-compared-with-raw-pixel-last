"""
Backbone of ResNet
"""
import torch.nn as nn
import torchvision.models as models
from utils.srm import SRMLayer

def build_ResNet101(pretrained):
    return ResNet_conv_body('resnet101', pretrained)


def build_ResNet50(pretrained):
    return ResNet_conv_body('resnet50', pretrained)


class ResNet_conv_body(nn.Module):
    def __init__(self, net, pretrained):
        super(ResNet_conv_body, self).__init__()
        if net == 'resnet50':
            model = models.resnet50(pretrained)
        elif net == 'resnet101':
            model = models.resnet101(pretrained)

        self.C1 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.C2 = model.layer1
        self.C3 = model.layer2
        self.C4 = model.layer3
        self.C5 = model.layer4

    def stages(self):
        return [self.C1, self.C2, self.C3, self.C4, self.C5]


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False


class ResNet(nn.Module):
    def __init__(self, net, pretrained, use_srm = False):
        super(ResNet, self).__init__()
        self.use_srm = use_srm
        
        if self.use_srm:
            self.srm = SRMLayer()
        if net == 'resnet50':
            self.model = models.resnet50(pretrained)
        elif net == 'resnet101':
            self.model = models.resnet101(pretrained)
        elif net == 'resnet152':
            self.model = models.resnet152(pretrained)
        elif net == 'resnet18':
            self.model = models.resnet18(pretrained)
        self.exfc = nn.Sequential(nn.ReLU(True), nn.Linear(1000, 5))
        
    def forward(self, x):
        if self.use_srm:
            x = self.srm(x)
        x = self.model(x)
        x = self.exfc(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, pretrained):
        super(DenseNet, self).__init__()
        self.model = models.densenet169(pretrained)
        self.exfc = nn.Sequential(nn.ReLU(True), nn.Linear(1000, 29))

    def forward(self, x):
        x = self.model(x)
        return self.exfc(x)


if __name__ == '__main__':
    net = ResNet('resnet50', True)
    net.cuda()
    from torchsummary import summary
    summary(net, (3, 224, 224))
    print(net)