import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class Dataset1(Dataset):
    def __init__(self,data_path,data_divider):
        data=np.array(pd.read_csv(data_path))
        n=data.shape[1]-2 # Number of elements
        X1=data[:,:n]/100 #Scaling the composition percentage by 100
        X2=(data[:,n]/1000).reshape(data.shape[0],1) #Scaling the temperature by 1000
        X=np.append(X1,X2,axis=1)
        Y=(data[:,(n+1)]/data_divider).reshape(data.shape[0],1) # Divide the label based on the data to model
        

        self.len=X.shape[0]
        self.x_data=torch.FloatTensor(X)
        self.y_data=torch.FloatTensor(Y)
        
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
         return self.len
