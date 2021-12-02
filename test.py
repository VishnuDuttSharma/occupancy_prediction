__author__ = "Vishnu Dutt Sharma"

import argparse
import numpy as np

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision import transforms

from tqdm import tqdm
from dataloader import OccMapDataset
from models import UNet

from torchvision.utils import make_grid, save_image

## Setting random seeds
torch.manual_seed(1234)
import random
random.seed(1234)
np.random.seed(1234)

from train import Solver

##########################################################################################
##########################################################################################
##########################################################################################

def parge_arguments():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=128, help='Batch size')
    parser.add_argument('--model_path', '-m', dest='model_path', type=str, default=None, help='Model .pth file location')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parge_arguments()
    
    # Defining transform
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float)
            ])
    # load the data
    test_set = OccMapDataset(filename='./description_ang0.csv', transform=transform, mode='test')
    
    test_size = len(test_set)

    print(f'Test data size: {test_size}')

    # data loader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    # model
    net = UNet(n_channels=1, n_classes=1, bilinear=True)

    # load the model
    model_path = args.model_path
    
    solver = Solver(net, optimizer='sgd', loss_fn='mse', lr=0.01, max_epoch=1, 
                    verbose=True, save_best=True, early_stop=5, 
                    outfile=model_path, save_full=True, scale=1.0)
    

    solver.net = torch.load(model_path)

    ## Plottig results
    for data in test_loader:
        images = data['input image']
        labels = data['target image']

        # placing data on device
        images = images.to(solver.device)
        labels = labels.to(solver.device)

        # We don't need gradients here
        with torch.no_grad():
            # forward propagation 
            preds = solver.net(images)
        break
    
    num_examples = 5
    image_path = model_path.replace('.pth', '.png')
    save_image(make_grid(torch.cat([images[:num_examples], labels[:num_examples], preds[:num_examples]], axis=0).cpu(), nrow=num_examples), image_path, normalize=True )
    
    print('Done.')
