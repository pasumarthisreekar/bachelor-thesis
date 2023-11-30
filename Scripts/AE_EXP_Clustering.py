#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[2])

from sklearn.model_selection import train_test_split
from os.path import expanduser
from os.path import join

import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import torch



# Custom Modules
from cli import get_args

import Regularizers_Geod as Regs
import utilities as utils
import Models as M
import geodist as gd


# In[2]:


print(torch.__version__)
print(torch.cuda.is_available())


# In[3]:


torch.set_default_dtype(torch.float64)
# torch.set_deterministic(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current cuda device: ", torch.cuda.get_device_name(0))


# In[5]:


csvs = ['Pendigits']
datasets = [f'GMM_{csv}' for csv in csvs] 

for data in datasets:
    print(data)


# In[ ]:


python_seed, np_seed, torch_seed = 0, 1, 2
verbose=True
folder = 'ClassificationClustering'

for factor in [int(sys.argv[1])]:
    for dataset in datasets:

        fp = join('.', folder, 'Data', f'{dataset}.csv')
        df = pd.read_csv(fp, header=None)

        X = df.iloc[:, :-1].to_numpy()

        input_dim = X.shape[1]
        encoding_dim = int(round(X.shape[1]/factor))
        factor_1 = factor - 1
        test_dim = int(round(X.shape[1]/factor_1))
        
            
        if factor > 2 and encoding_dim == test_dim:
            continue

        #addition/change of geod
        n_neighbors = max(5, round(0.01*len(X)))
        geod = gd.create_geodesic_matrix(X,n_neighbors)

        # Create tensors and move to GPU
        X_gpu = torch.from_numpy(X)
        X_gpu = X_gpu.to(device)

        geod = torch.from_numpy(geod)
        geod = geod.to(device)
        #removed reg2log: new regualrizer in reg1log
        for reg in ['noreg', 'reg1log']:
            Regularizer = Regs.get_regularizer(reg)

            torch.manual_seed(torch_seed)
            random.seed(python_seed)
            np.random.seed(np_seed)

            ae = M.DeepAutoEncoder(layers=[input_dim, 100, 100, 100, encoding_dim])

            ae.pretrain(X_gpu, epochs=[64, 64, 64, 64], device=device, lr=5e-4, verbose=verbose, train=not True)
            ae.to(device)
            M.unfreeze_layers(ae, verbose)

            lr = 1e-3
            e_lr, d_lr, = lr, lr

            optimizer = torch.optim.RMSprop(
                [{'params': ae.encoder.parameters(), 'lr': e_lr},
                 {'params': ae.decoder.parameters(), 'lr': d_lr}],)
            #     weight_decay=1e-9)

            model = M.CustomModelWrapper(ae, loss_fn=F.mse_loss, loss_coeff=1e0, reg_fn=Regularizer, reg_coeff=1e0)
            model.train(optimizer, X_gpu, geod, epochs=4*250, verbose=verbose, X_val=None, X_geod_val=None, batch=32)

            model._model.eval()
            _, Xl = model(X_gpu)
            Xl = Xl.cpu().detach().numpy()

            name = f'AE_{reg}_{dataset}'.upper()
            np.save(join('.', folder, 'Originals', f'{name}_XL_1000_1e3_{factor}'.upper()), Xl)