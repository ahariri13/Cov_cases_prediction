import math
import networkx as nx
import numpy as np
import torch

import os
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
__all__ = ['FCGPU']

class FCMuonsGPU(object):

    def __init__(self, name, sub):
        super(FCGPU, self).__init__()
        
       # direc=os.listdir(lista)

        self.all=torch.load(name)
               # self.batch=dgl.batch()
        #self.graphs=self.all[0][:sub]
        #self.labels=self.all[1]['glabel'].tolist()[:sub]
          #  return a,torch.tensor(np.array(b),dtype=torch.long).to(device)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self,idx):

        return self.all[idx]#self.graphs[idx], self.labels[idx]

