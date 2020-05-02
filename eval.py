# Let us look at how the network performs on the test dataset.
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

def eval(testloader, model,criterion,evaluate=False,data_divider=None, avg=None):
    use_gpu=False
    MSE = 0.0
    REL_ERR = 0.0
    RMSE = 0.0
    R2 = 0.0
    criterion1=torch.nn.L1Loss()
    with torch.no_grad():
        for data in tqdm(testloader,position=0):
            # get the inputs
            inputs, labels = data
            if use_gpu:
                inputs=Variable(inputs).cuda()
                labels=Variable(labels).cuda()
                labels=labels.to(device='cuda')
            else:
                inputs=Variable(inputs).cpu()
                labels=Variable(labels).cpu()
                labels=labels.to(device='cpu')


            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            rmse = torch.sqrt(loss)
            loss1=  criterion1(outputs, labels)
            # addup loss
            op=labels.squeeze()
            MSE += loss.item()
            if evaluate==True:
                REL_ERR += (loss1.item()/op.cpu().numpy())
                RMSE += rmse.item()            
                R2 += np.square(op.cpu().numpy()*data_divider-avg)
    if evaluate==True:
        mse_baseline=R2/len(testloader)
        mse=MSE/len(testloader)
        R2=1-(mse/mse_baseline)    
        print('The loss MSE: ',MSE/len(testloader),'RMSE: '\
              ,RMSE/len(testloader),'Rel err: ',REL_ERR/len(testloader),\
              'R value ',R2)
    else:
        print('Validation loss: %f' %(MSE / (len(testloader))))
    return MSE / len(testloader)
