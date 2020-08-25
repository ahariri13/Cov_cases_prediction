import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn as nn
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class GraphConvolution(Module):
    def __init__(self,bs, in_features, out_features, dropout=0., act=F.leaky_relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        #self.rand = torch.randn(200, in_features, out_features)
        self.rand=Parameter(torch.FloatTensor(bs,in_features, out_features),requires_grad=True).to(device)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):   
        input = F.dropout(input, self.dropout, self.training)  ## X.dropout()
        #self.rand=self.rand.to(self.device)
        support = torch.bmm(input, self.rand)  ## X.W
        support=F.dropout(support)
        #support = nn.Linear(input, self.weight)  ## X.W
        support=self.act(support)
        output = torch.bmm(adj,support) ##  A.X.W
        output = self.act(output) ## ReLU(A.X.W)
        return output         

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



"""
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn as nn

class GraphConvolution(Module):

    def __init__(self, bs, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        self.rand = torch.randn(bs, in_features, out_features,requires_grad=True)
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):   
        input = F.dropout(input, self.dropout, self.training)  ## X.dropout()
        support = torch.bmm(input, self.rand)  ## X.W
        #support = nn.Linear(input, self.weight)  ## X.W
        output = torch.bmm(adj, support) ##  A.X.W
        output = self.act(output) ## ReLU(A.X.W)
        return output         

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
"""