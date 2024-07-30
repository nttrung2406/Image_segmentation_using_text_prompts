from torch.utils.data import Dataset
import json
import pandas as pd

from torchvision.transforms.transforms import Resize
from torchvision import transforms

from PIL import Image
import os
from torch import Tensor
from datasets.utils import blend_image_segmentation
import torch.nn.functional as F
import random
import torch

class Human_MADS_Dataset(Dataset):
    def __init__(self, data_path = r'datasets\Human_MADS',\
                 split = 'train',image_size = 352, mask = 'text_and_crop_blur_highlight352' ,\
                      aug =None, with_class_label=False ) -> None:
        super().__init__()

        #Get directory of Image and Mask
        self.data_path = os.path.join(data_path, r'segmentation_full_body_mads_dataset_1192_img\segmentation_full_body_mads_dataset_1192_img')

        #Load all file name from csv file
        with open(os.path.join(data_path, 'df.csv')) as file:
            self.file_name_list = pd.read_csv(file, delimiter=',')

        #Split data for train and valdiate with ratio 0.7 : 0.3
        self.train_length = int(0.7*len(self.file_name_list))

        self.file_name_list = self.file_name_list.iloc[:self.train_length] if split =='train' \
            else self.file_name_list.iloc[self.train_length:]
        
        self.data_length = len(self.file_name_list)
        
        #This Dataset has only one class
        self.class_name = 'human'
        
        self.with_class_label = with_class_label

        #Transformer
        self.transform =transforms.Compose([
            transforms.Resize((image_size,image_size)), 
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
        ])

        self.mask_transform =transforms.Compose([
            transforms.Resize((image_size,image_size)), 
            transforms.ToTensor()
        ])

        self.mask = mask




    def __getitem__(self, index: int): #-> tuple[tuple[Tensor, str, Tensor], tuple[Tensor,Tensor]]:
        
        #Random Index of Support Image
        support_idx =...
        while True:
            support_idx = random.randint(0, self.data_length-1)
            if support_idx != index: break

        #Load Querry Image and Mask
        query_img_name, query_mask_name = self.file_name_list.iloc[index]['images'], self.file_name_list.iloc[index]['masks']

        query_img = Image.open(os.path.join(self.data_path, query_img_name))
        query_img =self.transform(query_img)

        query_mask = Image.open(os.path.join(self.data_path, query_mask_name))

        #Since mask is a RGBA Image and its alpha channel isn't used, we take one of three remain channels for binary mask 
        
        query_mask = self.mask_transform(query_mask.getchannel(1))
        query_mask = (query_mask > 0.5).float().squeeze(0)

        #Load Support Image and Mask
        support_img_name , support_mask_name = self.file_name_list.iloc[support_idx]['images'], \
            self.file_name_list.iloc[support_idx]['masks']
        
        support_img = Image.open(os.path.join(self.data_path, support_img_name))
        support_img = self.transform(support_img)

        support_mask = Image.open(os.path.join(self.data_path, support_mask_name))
        support_mask =self.mask_transform(support_mask.getchannel(1))
        support_mask = (support_mask > 0.5).float().squeeze(0)


        mask=self.mask
        if mask == 'separate':
            supp = (support_img, support_mask)   
        else:
            if mask.startswith('text_and_'):
                mask = mask[9:]
                label_add = [self.class_name]
            else:
                label_add = []

            supp = label_add + blend_image_segmentation(support_img, support_mask, mode=mask)

        supp = label_add + blend_image_segmentation(support_img,support_mask , mode= mask)
        
        if self.with_class_label:
            label = (torch.zeros(0), 0,)
        else:
            label = (torch.zeros(0), )

        return (query_img,) + tuple(supp) ,  (query_mask.unsqueeze(0), ) + label


    
    def __len__(self) -> int:
        return self.data_length
    
a = Human_MADS_Dataset()

