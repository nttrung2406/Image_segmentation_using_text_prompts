import os

from torch.utils.data import Dataset, DataLoader
import torch

from torchvision.transforms.transforms import Resize
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import json
from datasets.utils import blend_image_segmentation
import torch.nn.functional as F
import random

AGCClasses = ['NormalLeaf', 'AlternariaBoltch', 'AppleScab', 'BlackRot', 'CedarAppleRust', 'GrapeBlackrot', 'GrapeEsca', 'StrawberryLeaf']

class ArgiculturalDataset(Dataset):
    def __init__(self,split = 'train', image_size = 352, mask = 'text_and_', aug = None) -> None:
        self.data_dir = os.getcwd()+ '\\datasets\\argiculturedataset'
       
        self.mask=mask

        self.image_size = image_size
        self.transform =transforms.Compose([
            transforms.Resize((image_size,image_size)), 
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
        ])

        self.split = 'train' if split == 'train' else 'test'
        with open(os.path.join(self.data_dir, self.split + '.json'), "r") as f:
            self.image_dict = json.load(f)

    def __len__(self):
        length = 0
        for item in self.image_dict.keys():
            length += len(self.image_dict[item])
        return length

    def __getitem__(self, index):
        class_name = AGCClasses[index % len(AGCClasses)]

        img_path = os.path.join(self.data_dir, 'images')

        img_list = self.image_dict[class_name]
        img_len = len(img_list)
        # Load Query Image
        query_name = img_list[random.randint(0, img_len-1)]
        query_img = Image.open(os.path.join(img_path, query_name)).convert('RGB')
        query_img = self.transform(query_img)

        query_mask = self.load_mask(query_name).float()
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:],mode='nearest').squeeze()
        
        #Load Support Image
        support_name = ...
        while True:
            support_name = img_list[random.randint(0,img_len-1)]
            if support_name != query_name: break

        support_img = Image.open(os.path.join(img_path, support_name)).convert('RGB')
        support_img = self.transform(support_img)

        support_mask = self.load_mask(support_name).float()
        support_mask = F.interpolate(support_mask.unsqueeze(0).unsqueeze(0).float(), support_img.size()[-2:], mode='nearest').squeeze()

        mask=self.mask
        if (mask.startswith('text_and_')):
            mask = mask[9:]
            label_add = [class_name]
        else:
            raise Exception('Mask %s is not supported'.format(self.mask))

        supp = label_add + blend_image_segmentation(support_img,support_mask , mode= mask)
        
        label = (torch.zeros(0), )
        
        return (query_img,) + tuple(supp) ,  (query_mask.unsqueeze(0), ) + label

    def load_mask(self, mask_name):
        mask_path = os.path.join(os.path.join(self.data_dir, 'masks'), mask_name)
        mask = torch.tensor(np.array(Image.open(mask_path.replace('.jpg','.png'))))
        return mask