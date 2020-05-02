#To test for single value using the trained model
import numpy as np
import torch
from model import Net


a=np.array([5.5,9,8,0,0,0,0.6,0.1,3.2,0.8,0,9.5,63.3,25])
ch='k'
checkpoint='checkpoint\Best_model_'+ch+'.pth'
if ch=='k':
    print('Thermal conductivity prediction..........')
    data_divider=100
if ch=='density':
    print('Density prediction..........')
    data_divider=10
if ch=='cp':
    print('Heat capacity prediction..........')
    data_divider=1000
    
    
n=len(a)-1
new_net = Net(n_feature=n+1, n_hidden1=20, n_hidden2=5,n_output=1)  
new_net.load_state_dict(torch.load(checkpoint))
a=a/100 #Final with hari
a[-1]=a[-1]/10
with torch.no_grad():
    inp=a.reshape(1,n+1)
    input = torch.FloatTensor(inp)
    out = new_net(input).detach().numpy()
    print('The predicted output is ',out[0]*data_divider)