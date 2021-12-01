__author__ = "Vishnu Dutt Sharma"

import argparse
import numpy as np

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from tqdm import tqdm
from dataloader import OccMapDataset
from models import UNet

## Setting random seeds
torch.manual_seed(1234)
import random
random.seed(1234)
np.random.seed(1234)

class Solver(object):
    def __init__(self, net, optimizer='sgd', loss_fn='mse', lr=0.1, max_epoch=10, verbose=True, save_best=True, early_stop=None, outfile='./models/some_net.pth', save_full=True):
        # Your code 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = net.to(self.device)
        
        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_fn == 'kl':
            raise NotImplementedError
        else: # Wasserstien
            raise NotImplementedError
        
        if optimizer == 'sgd': 
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        else:
            self.optimizer = optim.Adadelta(self.net.parameters(), lr=lr)
        
        if early_stop is not None:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)
        
        self.max_epoch = max_epoch
        self.verbose = verbose
        self.early_stop = early_stop
        self.outfile = outfile
        self.save_full = True
        
        self.writer = SummaryWriter('./logs/')

    def train(self, train_loader, valid_loader=None):
        """Function to train the model

        Parameters
        ----------
            train_loader: Training data loader
            valid_loader: Validation data loader
        
        Returns
        -------
            None
        """
        # Your code 
        ## Initialing minimum loss with a large value
        min_valid_loss = np.inf
        ## Indicator for early stopping
        stopping = False

        ## Lists to save training and validation loss at each epoch
        training_loss_list = []
        validation_loss_list = []

        ## Patience counter for early stopping
        early_stop_count = self.early_stop
        
        ## Global counter
        global_count = 0
        
        ## Iterating over each epoch
        for ep in range(self.max_epoch):  
            ## Initializing episodic loss
            ep_loss = 0.0
            ## Iterating through batches fo data
            for idx, data in enumerate(train_loader):
                # Getting the inputs; data is a list of [inputs, labels]
                images = data['input image']
                labels = data['target image']
                
                # placing data on device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward propagation 
                preds = self.net(images)
                
                # calculating loss
                loss = self.criterion(preds, labels)
                
                # backprop
                loss.backward()
                self.optimizer.step()

                # Getting loss
                ep_loss += loss.item()

                # printing progress
                if self.verbose and ((idx+1)% 20 == 0):    # print every 20 mini-batches
                    valid_loss  = self.test(valid_loader)
                    print(f'Episode: {ep+1}, Iteation: {idx+1}, Validation loss: {valid_loss}')
                    self.writer.add_scalar("GlobalLoss/valid", valid_loss, global_count)
                    self.net.train();
                
                # Update global iteration count
                global_count += 1
            
            
            ## Getting the average loss 
            training_loss = ep_loss/len(train_loader)
            
            ## Saving the episodic loss
            training_loss_list.append(training_loss)

            if self.verbose:
                print(f'End of Episode {ep+1}, Training loss: {training_loss}')
            
            self.writer.add_scalar("Loss/train", training_loss, ep+1)
            self.writer.add_scalar("GlobalLoss/train", training_loss, global_count)

            ## Calculaing the validation loss for this epoch
            valid_loss = self.test(valid_loader)
            ## Moving model back to training model
            self.net.train();

            ## Saving the validation loss for this epoch
            validation_loss_list.append(valid_loss)
            
            ## Printing progress
            if self.verbose:
                print(f'Validation loss: {valid_loss}')
            
            self.scheduler.step(valid_loss)
            self.writer.add_scalar("Loss/valid", valid_loss, ep+1)
            self.writer.add_scalar("GlobalLoss/valid", valid_loss, global_count)
            
            ## If current loss is less than minimum loss so far, update it and save model
            if valid_loss <= min_valid_loss:
                min_valid_loss = valid_loss
                
                ## Saving model or model state_dict
                if self.save_full:
                    torch.save(solver.net, self.outfile)
                else:
                    torch.save(solver.net.state_dict(), self.outfile)
                
                if self.verbose:
                    print('Saving model')

                ## If early_stopping is enabled, then reset the patience
                if self.early_stop is not None:
                    early_stop_count = self.early_stop

            elif self.early_stop is not None: # if current validation loss is larger than the minimum loss so far, reduce patience
                early_stop_count -= 1
                ## If patience is 0, stop training
                if early_stop_count == 0:
                    stopping = True

            if stopping:
                print(f'Stoppping early')
                break
            
            
            self.writer.flush()
        
        self.write.close()
        print('Training completed')
        

        #### Plotting trainig and test curves
#         plt.plot(np.arange(1,len(training_loss_list)+1), training_loss_list, 'b', label='Training')
#         if valid_loader is not None:
#             plt.plot(np.arange(1,len(validation_loss_list)+1), validation_loss_list, 'g', label='Validation')
        
#         plt.xlabel('#Epochs')
#         plt.ylabel('Loss (Cross-Ent)')
#         plt.legend(loc="upper right")

    def test(self, loader):
        """Function to test the model

        Parameters
        ----------
            loader: Validation or test loader

        Returns
        -------
            float: loss
            float: accuracy
        """
        
#         ## Placeholder to save predictions and GT labels
#         preds_list = []
#         label_list = []
        
        ## Initlaizing the loss
        test_loss  = 0
        
        ## Moving model to eval model
        self.net.eval();
        
        for data in loader:
            # Getting the inputs; data is a list of [inputs, labels]
            images = data['input image']
            labels = data['target image']

            # placing data on device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # We don't need gradients here
            with torch.no_grad():
                # forward propagation 
                preds = self.net(images)
            
            # calculating loss
            loss = self.criterion(preds, labels)

#             # Saving the predictions and labels
#             preds_list.append(preds.cpu().data.argmax(1).numpy())
#             label_list.append(labels.cpu().data.numpy())

            # Adding the batch loss to the total loss
            test_loss += loss.item()

#         # Converting lists to arrays for easier processing
#         preds_np = np.concatenate(preds_list)
#         label_arr = np.concatenate(label_list)

#         # Calculating test accuracy
#         test_acc = 100*(preds_np == label_arr).sum()/len(label_arr)
        
        # Calculating average loss
        test_loss_norm = test_loss/len(loader)

        return test_loss_norm#, test_acc
    
##########################################################################################
##########################################################################################
##########################################################################################

def parge_arguments():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', dest='ep', metavar='E', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=128, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parge_arguments()
    
    # Defining transform
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float)
            ])
    # load the data
    full_set = OccMapDataset(filename='./description_ang0.csv', transform=transform)
    full_size = len(full_set)

    train_size = int((100 - 2*args.val)/100. * full_size)
    valid_size = int((100 - args.val)/100. * full_size) - train_size
    test_size = full_size - train_size - valid_size

    print(f'Data sizes:\nTrain: {train_size}\nValid: {valid_size}\nTest: {test_size}')

    train_val_set, test_set = torch.utils.data.random_split(full_set, [train_size+valid_size, test_size])
    train_set, valid_set = torch.utils.data.random_split(train_val_set, [train_size, valid_size])

    # data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=2*args.batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=2*args.batch_size, shuffle=False, num_workers=2)
    
    # model
    net = UNet(n_channels=1, n_classes=1, bilinear=True)

    # train the model
    model_path = f"./saved_models/{'sgd'}_LR_{args.lr}_epoch_{args.ep}.pth"
    
    solver = Solver(net, optimizer='sgd', lr=args.lr, max_epoch=args.ep, 
                    verbose=True, save_best=True, early_stop=5, 
                    outfile=model_path, save_full=True)
    if not args.load:
        solver.train(train_loader, valid_loader)
    else:
        print(f'Loading pre-trained model from {model_path}')
        # solver.net.load_state_dict(torch.load(model_path))
        solver.net = torch.load(model_path)

    test_loss = solver.test(loader=test_loader)
    print(f'Test loss: {test_loss}')
    
    print('Done.')
