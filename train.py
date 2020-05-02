# Train the network
# ^^^^^^^^^^^^^^^^^^^^
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import torch

def train(epoch, trainloader,model, optimizer, criterion):
    running_loss = 0.0
    use_gpu=False
    for i, data in enumerate(tqdm(trainloader,position=0), 0):
        # get the inputs
        inputs, labels = data
        #print(inputs.shape,labels.shape)
        if torch.isnan(labels)==1:
            continue

        if use_gpu:
            inputs=Variable(inputs).cuda()
            labels=Variable(labels).cuda()
            labels=labels.to(device='cuda')
        else:
            inputs=Variable(inputs).cpu()
            labels=Variable(labels).cpu()
            labels=labels.to(device='cpu')

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # addup loss
        running_loss += loss.item()
    print('Training loss: %f' %(running_loss / (len(trainloader))))
    return running_loss / (len(trainloader))
    