#Dataset.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data import Dataset1
from model import Net
from train import train
from eval import eval
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import time
import os
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='ANN training parameters')
parser.add_argument('--data_path',help='Address of the data to train in csv format')
parser.add_argument('--ch',help='data to train k- thermal conductivity, density- density, cp- heat capacity')
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_elements',type=int, default=13)
parser.add_argument('--save_cp', default=False, help='Saving checkpoint')
parser.add_argument('--load_cp', default=False, help='Checkpoint to load')
args = parser.parse_args()

torch.manual_seed(1)
#Choosing which data to train for
if args.ch=='k':
    print('Thermal conductivity prediction..........')
    data_divider=100
    avg=19.74
if args.ch=='density':
    print('Density prediction..........')
    data_divider=10
    avg=8.41
if args.ch=='cp':
    print('Heat capacity prediction..........')
    data_divider=1000
    avg=529.47
    
shuffle_dataset = True
dataset=Dataset1(args.data_path,data_divider)
dataset_size = dataset.len
print("dataset has length=",dataset_size)

indices = list(range(dataset_size))
#Split the data into train, validation and test data
split1 = int(np.floor(0.8 * dataset_size))
split2=int(np.floor(0.9 * dataset_size))

if shuffle_dataset :
    np.random.shuffle(indices)
train_indices1,train_indices2, test_indices = indices[:split1], indices[split1:split2], indices[split2:len(indices)]
#Sampling the data for training, validating and testing
train_sampler1 = SubsetRandomSampler(train_indices1)
train_sampler2 = SubsetRandomSampler(train_indices2) 
test_sampler = SubsetRandomSampler(test_indices)
train_loader1 = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler1)
train_loader2 = DataLoader(dataset, batch_size=args.batch_size,sampler=train_sampler2)                                          
test_loader = DataLoader(dataset, batch_size=args.batch_size,sampler=test_sampler)
                                                
												
#Training
net = Net(n_feature=args.num_elements+1, n_hidden1=20, n_hidden2=5,n_output=1)     # define the network
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5,betas=(0.9,0.999))
criterion = torch.nn.MSELoss()  # this is for regression mean squared loss
if args.load_cp: # Start training with a checkpoint
    net.load_state_dict(torch.load(args.load_cp, map_location=device))
    print('Loading the checkpoint ',args.load_cp)
print('Start Training')
if args.save_cp==True: # To store the checkpoint
    if not os.path.exists('./models'):
        os.mkdir('./models')

training_losses = []
testing_losses = []
start=time.clock()

for epoch in range(args.num_epochs):  # loop over the dataset multiple times
    print('EPOCH ', epoch + 1)
    train_loss = train(epoch, train_loader1,net, optimizer, criterion)
    test_loss = eval(train_loader2, net,criterion)
    if args.save_cp==True:
        torch.save(net.state_dict(), './models/model--'+str(epoch)+'.pth')    
    training_losses.append(train_loss)
    testing_losses.append(test_loss)

print('Finished Training')
print('Time taken for training is ',time.clock()-start)
print('Evaluating with the test data ')
eval(train_loader2, net,criterion,True,data_divider,avg)

#Plot training and validation loss curve
plt.plot(range(len(training_losses)),training_losses)
plt.title('Training loss curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
plt.plot(range(len(testing_losses)),testing_losses)
plt.title('Validation loss curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()