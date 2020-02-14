#import os
#os.chdir("D:\Matching\Code_kdd_matching\python_flow")

import time
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import torch.optim as optim
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from torch.distributions import MultivariateNormal
from torch.distributions import Uniform
from torch.distributions.bernoulli import Bernoulli

from nf.flows import *
import random
import pandas as pd


parser = ArgumentParser(description='manual to this script')
parser.add_argument('--num', type=int, default = 1)

args = parser.parse_args()
num = args.num





torch.set_num_threads(1)


def get_neighbors(focal_row, match_rows):
	m=len(match_rows)
	dist = (focal_row.expand(m) - match_rows).pow(2)
    
	matched_index = dist.argmin().item()
	return matched_index

class flow_model(nn.Module):
    
    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flows = nn.ModuleList(flows)
    
    def forward(self, x):
        log_det = torch.zeros_like(x[:,0]) # log_det.shape = [n]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        
        return z, prior_logprob, log_det # z.shape=[n,d], prior_logprob.shape = log_det.shape = [n]


def main(i):
    device = torch.device("cpu")
    print("The device is", device)
    

    
    name = "data/data_all_"+str(i)+".csv"
    data = pd.read_csv(name)
    
    data_list=data.values.tolist()
    data_array=np.array(data_list)
    data_torch=torch.from_numpy(data_array)
    (x, prob_w, w, y, truth)=(data_torch[:,0:8].float(),data_torch[:,8].unsqueeze(1).float(),data_torch[:,9].float(),data_torch[:,10].unsqueeze(1).float(),data_torch[:,11].float())
    
    data_all = (x, w, prob_w, y)
    data_all = (data_all[0].to(device), data_all[1].to(device), data_all[2].to(device), data_all[3].to(device))

    y1_index=(data_all[1]==1).nonzero().squeeze() # get the index of w=1
    y0_index=(data_all[1]==0).nonzero().squeeze()
    data_1=(data_all[0][y1_index]) # get the data of w=1
    data_0=(data_all[0][y0_index])
    print("data_1_all number is",len(y1_index))
    print("data_0_all number is",len(y0_index))
    


    (input_train, input_eval) = train_test_split(data_1.cpu().numpy(), test_size=0.20, random_state=4)    
    data_1_train=(torch.from_numpy(input_train).to(device))    
    data_1_eval=(torch.from_numpy(input_eval).to(device))
    
    (input_train, input_eval) = train_test_split(data_0.cpu().numpy(), test_size=0.20, random_state=4)    
    data_0_train=(torch.from_numpy(input_train).to(device))    
    data_0_eval=(torch.from_numpy(input_eval).to(device))
    
    del input_train, input_eval


    data_0= Data.TensorDataset(data_0_train)
    loader_0 = Data.DataLoader(data_0, batch_size = 32, shuffle=True)
    print('Number of train_0: ', len(data_0))

    data_1= Data.TensorDataset(data_1_train)
    loader_1 = Data.DataLoader(data_1, batch_size = 32, shuffle=True)
    print('Number of train_1: ', len(data_1))
    
    
    # training begins
    prior = MultivariateNormal(torch.zeros(d).to(device), torch.eye(d).to(device))
      
    flows_0 = [Planar(dim=d), Planar(dim=d),Planar(dim=d),Planar(dim=d)]
    model_0 = flow_model(prior, flows_0)# define the model to get learn p(x|w=0)    
    model_0.to(device)
    optimizer_0 = optim.Adam(model_0.parameters(), lr=0.001)

    for name, parameters in model_0.named_parameters():
        if parameters.requires_grad:
            print(name,parameters)
    
        
    logprob_eval=[]
    logprob_train=[]

    j=0
    start_epoch0=time.time()
    for epoch in range(300):

        train_logprob  = 0
        train_prior = 0
        train_log_det = 0
        train_steps = 0
        for i,data in enumerate(loader_0):
            optimizer_0.zero_grad()
    
            z, prior_logprob, log_det = model_0(data[0]) # z.shape=[n,d], log_det.shape=prior_logprob.shape=[n]
            logprob = prior_logprob + log_det
            loss = -torch.mean(logprob)
            
            #accumulate
            train_logprob += logprob.mean().item()
            train_prior += prior_logprob.mean().item()
            train_log_det += log_det.mean().item()
            train_steps += 1
    
            # backward and update
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model_0.parameters(), 3, norm_type=2)
    
            optimizer_0.step()
    
        train_logprob /= train_steps
        train_prior /= train_steps
        train_log_det /= train_steps
    
        print('Epoch_control:[%d]   train_logprob:%.5f   train_prior:%.5f  train_log_det:%.5f' %
                            (epoch, train_logprob, train_prior, train_log_det))
        if epoch==0:
            end_epoch0=time.time()
            print("Time for one epoch is", end_epoch0-start_epoch0)
        model_0.eval()
    
        with torch.no_grad():
            z, prior_logprob, log_det = model_0(data_0_eval)
            logprob = prior_logprob + log_det
            logprob_eval.append(logprob.mean().item())
    
#            z, prior_logprob, log_det = model_0(data_0_train)
#            logprob = prior_logprob + log_det
#            logprob_train.append(logprob.mean().item())
    
        locals()["model_0_"+str(j)]=model_0
        j+=1
        model_0.train()
    
    print('The logprob_eval of model_0 is', logprob_eval)
    j0=logprob_eval.index(max(logprob_eval))
    print('The best model_0 index is %d'%(j0))
    print('The best logprob_eval is', max(logprob_eval))
    a=max(logprob_eval)
    model_0=locals()["model_0_"+str(j0)]
    model_0.eval()
    



    flows_1 = [Planar(dim=d), Planar(dim=d),Planar(dim=d),Planar(dim=d)] 
    model_1 = flow_model(prior, flows_1)# define the model to get learn p(x|w=0)    
    model_1.to(device)
    optimizer_1 = optim.Adam(model_1.parameters(), lr=0.001)
    
    logprob_eval=[]
    logprob_train=[]
    j=0
    start_epoch0=time.time()
    for epoch in range(300):
        train_logprob  = 0
        train_prior = 0
        train_log_det = 0
        train_steps = 0
        for i,data in enumerate(loader_1):
            optimizer_1.zero_grad()
    
            z, prior_logprob, log_det = model_1(data[0]) # z.shape=[n,d], log_det.shape=prior_logprob.shape=[n]
            logprob = prior_logprob + log_det
            loss = -torch.mean(logprob)

            #accumulate
            train_logprob += logprob.mean().item()
            train_prior += prior_logprob.mean().item()
            train_log_det += log_det.mean().item()
            train_steps += 1
    
            # backward and update
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model_1.parameters(), 3, norm_type=2)
    
            optimizer_1.step()
    
        train_logprob /= train_steps
        train_prior /= train_steps
        train_log_det /= train_steps
    
        print('Epoch_treat:[%d]   train_logprob:%.5f   train_prior:%.5f  train_log_det:%.5f' %
                            (epoch, train_logprob, train_prior, train_log_det))
        if epoch==0:
            end_epoch0=time.time()
            print("Time for one epoch is", end_epoch0-start_epoch0)
        model_1.eval()
    
        with torch.no_grad():
            z, prior_logprob, log_det = model_1(data_1_eval)
            logprob = prior_logprob + log_det
            logprob_eval.append(logprob.mean().item())
    
        locals()["model_1_"+str(j)]=model_1
        j+=1
        model_1.train()
    
    print('The logprob_eval of model_1 is', logprob_eval)
    j=logprob_eval.index(max(logprob_eval))
    print('The best model_1 index is %d'%(j))
    print('The best logprob_eval is', max(logprob_eval))
    
    model_1=locals()["model_1_"+str(j)]
    model_1.eval()
    
    
    with torch.no_grad():
        
        z, prior_logprob, log_det = model_0(data_all[0])
        prob_0 = torch.exp(prior_logprob + log_det)
        
        z, prior_logprob, log_det = model_1(data_all[0])
        prob_1 = torch.exp(prior_logprob + log_det)
        
        prior_treat = len(y1_index)/(len(y1_index)+len(y0_index))
        
        prob_treat = prob_1 * prior_treat / (prob_1 * prior_treat + prob_0 * (1-prior_treat))
        
        prob_treat[prob_treat != prob_treat] = prior_treat
        
        loss = ((prob_treat - data_all[2].squeeze(1)).pow(2).mean()).item()
        print("The training loss is", loss)
        bias_pscore.append(loss)
    
        y1_index=(data_all[1]==1).nonzero().squeeze() # get the index of w=1
        y0_index=(data_all[1]==0).nonzero().squeeze()
        prob_treat_1 = prob_treat[y1_index]
        prob_treat_0 = prob_treat[y0_index]
    
        data_1=(data_all[0][y1_index], data_all[1][y1_index], data_all[2][y1_index], data_all[3][y1_index].squeeze(1)) # get the data of w=1
        data_0=(data_all[0][y0_index], data_all[1][y0_index], data_all[2][y0_index], data_all[3][y0_index].squeeze(1))
        
        ATT = (data_1[3].mean() - data_0[3].mean()).item()
        print("The average treatment effect for treated is", ATT)              
        ATT_base.append(ATT)
        
        matches=[]
        for i in range(len(prob_treat_1)):
            index = get_neighbors(prob_treat_1[i], prob_treat_0)
            matches.append(index)
        
        print("The number of treated units is",len(prob_treat_1))
        print("The number of matched units is",len(set(matches)))
        match_index = torch.tensor(matches)
        ITE = data_1[3] - data_0[3][match_index]
        ATT = ITE.mean().item()
        
        print("The average treatment effect for treated is", ATT)
        ATT_flow.append(ATT)
        
        ATT_true.append(truth[0].item())
        
       


d=8

ATT_base = []
ATT_flow = []
ATT_true = []

bias_pscore = []

start=time.time()
for i in range(1,101):
    main(i)


output=pd.DataFrame({"ATT_base":ATT_base,"ATT_flow":ATT_flow, \
                     "ATT_true":ATT_true,"bias_pscore":bias_pscore})
output.to_csv("output"+str(num)+".csv",index=False,sep=',')







ATT_base = np.array(ATT_base)
ATT_flow = np.array(ATT_flow)
ATT_true = np.array(ATT_true)


ATT_base_bias = (ATT_base-ATT_true).mean() 
ATT_base_sd = ATT_base.std() 
ATT_base_mae = (np.abs(ATT_base-ATT_true)).mean()
ATT_base_rmse = np.sqrt((np.square(ATT_base-ATT_true)).mean())

ATT_flow_bias = (ATT_flow-ATT_true).mean()
ATT_flow_sd = ATT_flow.std() 
ATT_flow_mae = (np.abs(ATT_flow-ATT_true)).mean()
ATT_flow_rmse = np.sqrt((np.square(ATT_flow-ATT_true)).mean())


print("ATT_base_bias is",ATT_base_bias)
print("ATT_base_sd is",ATT_base_sd)
print("ATT_base_mae is",ATT_base_mae)
print("ATT_base_rmse is",ATT_base_rmse)

print("ATT_flow_bias is",ATT_flow_bias)
print("ATT_flow_sd is",ATT_flow_sd)
print("ATT_flow_mae is",ATT_flow_mae)
print("ATT_flow_rmse is",ATT_flow_rmse)


name= "result"+".txt"
with open(name, 'w') as result:
    result.write('ATT_base_bias is %s \n' % ATT_base_bias)
    result.write('ATT_base_sd is %s \n' % ATT_base_sd)
    result.write('ATT_base_mae is %s \n' % ATT_base_mae)
    result.write('ATT_base_rmse is %s \n' % ATT_base_rmse)
    
    result.write('ATT_flow_bias is %s \n' % ATT_flow_bias)
    result.write('ATT_flow_sd is %s \n' % ATT_flow_sd)
    result.write('ATT_flow_mae is %s \n' % ATT_flow_mae)
    result.write('ATT_flow_rmse is %s \n' % ATT_flow_rmse)
    


end=time.time()
print("Time_use is", end-start)
    
    
    


