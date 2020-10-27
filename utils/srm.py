import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

class SRMLayer(nn.Module):
    def __init__(self):
        super(SRMLayer, self).__init__()
        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]
        filter1 = np.asarray(filter1, dtype=float) / q[0]
        filter2 = np.asarray(filter2, dtype=float) / q[1]
        filter3 = np.asarray(filter3, dtype=float) / q[2]
        filters = np.asarray([[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]) #shape=(3,3,5,5)
        #filters = np.transpose(filters, (2,3,1,0)) #shape=(5,5,3,3)
 
        initializer_srm = torch.tensor(filters).float()
        self.srm = nn.Conv2d(3, 3, (5, 5), padding=2)
        self.srm.weight = torch.nn.Parameter(initializer_srm)
        #print(self.srm.weight)
        
    def forward(self, x):    
        output = self.srm(x)
        return output

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


if __name__ == '__main__':
    srm = SRMLayer()
    srm.eval()
    img = cv2.imread('test.jpg')
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img)
    img = srm(transform(img).unsqueeze(0)).squeeze(0)
    
    img = (img+1)*128
    img = img.detach().numpy().astype(np.uint8).transpose(1, 2, 0)
    print(img.shape)
    cv2.imwrite('./srm_output.jpg', img)