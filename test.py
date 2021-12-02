__author__ = "Vishnu Dutt Sharma"

import argparse
import numpy as np

import torch
import torch.nn as nn
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

import matplotlib.pyplot as plt
import seaborn as sns
##########################################################################################
##########################################################################################
##########################################################################################

def parge_arguments():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=128, help='Batch size')
    parser.add_argument('--model_path', '-m', dest='model_path', type=str, default=None, help='Model .pth file location')
    parser.add_argument('--show', '-m', dest='model_path', type=str, default=None, help='Model .pth file location')
    
    return parser.parse_args()

def convert_to_occ(arr, low_prob_thresh=0.495, high_prob_thresh=0.505):
    occ_map = np.zeros(arr.shape, dtype=np.int) # default unknown
    occ_map[arr < low_prob_thresh] = -1 # free
    occ_map[arr > high_prob_thresh] = 1 # occupied
    
    return occ_map


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
    selver.net = solver.net.to(selver.device)
    
    input_list = []
    gt_list = []
    pred_list = []
    ## Plotting results
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
        
        input_list.append(images.cpu().data.numpy())
        gt_list.append(labels.cpu().data.numpy())
        pred_list.append(preds.cpu().data.numpy())
    
    
    o_inp = np.concatenate(input_list)
    o_gt = np.concatenate(gt_list)
    o_pred = np.concatenate(pred_list)
    
    inpainted = (o_inp == 0) & (o_pred != 0)
    sensed_cells = (o_inp != 0)
    
    inpainted_flat = inpainted.reshape(inpainted.shape[0], -1)
    sensed_cells_flat = sensed_cells.reshape(sensed_cells.shape[0],-1)
    
    occ_map_pred = convert_to_occ(o_pred)
    occ_map_gt = convert_to_occ(o_gt)
    match  = (occ_map_pred == occ_map_gt)
    match_flat = match.reshape(match.shape[0], -1)
    
    figs, axes = plt.subplots(1,2)
    frac_inp = 100%inpainted.reshape(inpainted.shape[0], -1).sum(axis=1)/(o_inp.shape[-1]*o_inp.shape[-2])
    sns.histplot(frac_inp, ax=axes[0]).set_title('% cells inpainted')
    acc = (match_flat * inpainted_flat).sum(axis=1)/inpainted_flat.sum(axis=1)
    sns.histplot(acc).set_title('Accuracy histogram')
    image_path = model_path.replace('.pth', '_METRICS.png')
    plt.savefig(image_path)
    

    num_examples = min(5, len(images))
    image_path = model_path.replace('.pth', '_TEST.png')
    save_image(make_grid(torch.cat([images[:num_examples], labels[:num_examples], preds[:num_examples]], axis=0).cpu(), nrow=num_examples), image_path, normalize=True )
    
    print('Done.')
