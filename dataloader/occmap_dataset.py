__author__ = "Vishnu Dutt Sharma"

import pandas as pd
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF

class OccMapDataset(data.Dataset):
    def __init__(self, filename='./description_ang0.csv', transform=None, input_dir='./inp_data/', target_dir='./gt_data/', mode='train', odds_to_prob=True, scale=10.):
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

        self.odds_to_prob = odds_to_prob
        self.scale = scale


    def __len__(self):
        # Your code 
        return len(self.filepaths)
    
    def __getitem__(self, index):
        # Your code
        filename = self.filepaths[index]
        inp_img = np.load(f'{self.input_dir}/{self.filepaths[index]}.npy')
        tgt_img = np.load(f'{self.target_dir}/{self.filepaths[index]}.npy')
        
        data_dict = {'input image': inp_img[:, :], 'target image': tgt_img[ :, :]}
        data_dict['input image'] *= self.scale
        data_dict['target image'] *= self.scale
        
        if self.odds_to_prob:
            o2p_func = lambda x: np.exp(x)/(1. + np.exp(x))
            data_dict['input image'] = o2p_func(data_dict['input image'])
            data_dict['target image'] = o2p_func(data_dict['target image'])

        if self.transform is not None:
            data_dict['input image'] = self.transform(data_dict['input image'])
            data_dict['target image'] = self.transform(data_dict['target image'])
            
        if np.random.random() > 0.5:
            data_dict['input image'] = TF.vflip(data_dict['input image'])
            data_dict['target image'] = TF.vflip(data_dict['target image'])
         
        return data_dict
