import os
import json
from torch.utils.data import Dataset, DataLoader
import numpy as np
from os import listdir
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import random
from pathlib import Path
from itertools import chain
# RGB

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

class get_dataset(Dataset):
    def __init__(self, data_root, choose_dirs, transform, mode='train'):
        super(get_dataset, self).__init__()
        self.data_root = data_root
        #self.choose_dirs = ['celeba'] + choose_dirs
        self.choose_dirs = choose_dirs
        self.mode = mode
        if mode == 'test':
            self.imgs, self.labels = self.make_test_dataset()
        else:
            self.imgs, self.labels = self.make_dataset()
        self.transform = transform

    def __getitem__(self, item):
    
        img_path = self.imgs[item]
        label = self.labels[item]
        
        img = Image.open(img_path).convert('RGB')
        
        return self.transform(img), label

    def __len__(self):
        return len(self.imgs)

    def make_test_dataset(self):

        imgs = []
        labels = []

        for i, dir in enumerate(self.choose_dirs):

            class_dir = os.path.join(self.data_root, dir)
            print(class_dir)

            fnames = listdir(class_dir)
            fnames.sort()
            fnames = fnames[-2000:]
            imgs += fnames
            labels += [1] * len(fnames)

        return imgs, labels

    def make_dataset(self):

        imgs = []
        labels = []
        
        for i, dir in enumerate(self.choose_dirs):
        
            class_dir = os.path.join(self.data_root, dir)
            print(class_dir)
            
            fnames = listdir(class_dir)
            fnames.sort()
            
            if self.mode == 'train':
                if len(fnames) < 20000:
                    fnames = fnames[:-1000]
                else:
                    fnames = fnames[:18000]
                
            elif self.mode == 'val':
                if len(fnames) < 20000:
                    fnames = fnames[-1000:]
                else:
                    fnames = fnames[18000:20000]    

            imgs += fnames
            labels += [i] * len(fnames)
            
        return imgs, labels


def get_dataset_loader(data_root, choose_dirs, batch_size, num_workers, mode='train'):
    if mode == 'train':
        transform = transforms.Compose([
            #transforms.ToPILImage(),
            #transforms.Resize((256, 256)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            #transforms.ToPILImage(),
            #transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])    

    dataset = get_dataset(data_root, choose_dirs, transform, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=shuffle)
    return data_loader


if __name__ == '__main__':

    data_root = '/data/data'
    choose_dirs = ['DR']
    dataset = get_dataset_loader(data_root, choose_dirs, 1, 0, 'val')
    print(len(dataset))
    for img, labels, in tqdm(dataset):
        print(img[0].size())
        print(labels)
        break
