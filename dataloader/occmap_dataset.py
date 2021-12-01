__author__ = "Vishnu Dutt Sharma"

import pandas as pd
import numpy as np
import torch.utils.data as data
from torchvision import transforms

class OccMapDataset(data.Dataset):
    def __init__(self, filename='./description_ang0.csv', transform=None, input_dir='./inp_data/', target_dir='./gt_data/', mode='train'):
        # Your code 
        df = pd.read_csv(filename)
        df = df[df['free_perc'] <= 80]
        
        df['FloorID'] = df['FloorName'].apply(lambda x: int(x[-3:]))
        if mode == 'train':
            df = df[df['FloorID'] <= 220]
        else:
            df = df[df['FloorID'] > 220]
            
        self.filepaths = df['Filename'].values
        
        self.transform = transform
        self.input_dir = input_dir
        self.target_dir = target_dir

    def __len__(self):
        # Your code 
        return len(self.filepaths)
    
    def __getitem__(self, index):
        # Your code
        filename = self.filepaths[index]
        inp_img = np.load(f'{self.input_dir}/{self.filepaths[index]}.npy')
        tgt_img = np.load(f'{self.target_dir}/{self.filepaths[index]}.npy')
        
        data_dict = {'input image': inp_img[:, :], 'target image': tgt_img[ :, :]}
        
        if self.transform is not None:
            data_dict['input image'] = self.transform(data_dict['input image'])
            data_dict['target image'] = self.transform(data_dict['target image'])
        
        return data_dict