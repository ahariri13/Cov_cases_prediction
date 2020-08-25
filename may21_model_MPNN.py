
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import GraphConvolution
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
import torch_geometric.nn as tnn
from layers import GraphConvolution
from torch.autograd import Variable
import numpy as np


class GCNModelVAE(nn.Module):
    def __init__(self, bs,input_feat_dim, hidden_dim1,hidden_dim2,dropout,out_channelsGated,num_layers):#input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()

        
        """
        CNN params: 
            
        """
        
        in_channels, out_channels = 3, 32
        kernel_size = (3,3)
        kernel_size = (3,3)
        self.cn1=nn.Conv2d(in_channels,out_channels,kernel_size)
    
        self.cn2=nn.Conv2d(out_channels,16,kernel_size)
        self.pool=nn.AvgPool2d(2,2)#nn.MaxPool2d(3,3)
        
        
        """
        GCN Params:
        """
        self.gc1 = GraphConvolution(bs,input_feat_dim, hidden_dim1, dropout, act=F.tanh)
        self.gc2 = GraphConvolution(bs,hidden_dim1, hidden_dim2, dropout, act=F.tanh) # mu 
        
        self.hid1=hidden_dim1
        self.hid2=hidden_dim2
        
        self.gtran1=nn.Linear(56,50)
        self.hun=nn.Linear(11,50)
        """
        GGN params:
        """
        self.ggn=tnn.GatedGraphConv(out_channelsGated,num_layers,aggr='mean')
        self.tr1=nn.Linear(21,50)
        self.tr2=nn.Linear(50,1)
        
        """
        Cases params:
        """
        
        self.proj=nn.Linear(352,hidden_dim2)

        """
        LSTM params:
        """

        self.bn1 = nn.BatchNorm1d(num_features=200)
        #self.ls=LSTM()
        self.slide=bs
        
        self.fc1=nn.Linear(500,256)
        self.fc2=nn.Linear(256,64)
        self.fc3=nn.Linear(64,48)
        
        self.hidden_size=4

        self.lstm2 = nn.LSTMCell(1, 4)
        self.linear2 = nn.Linear(4, 48)

    
    def trainIt(self,x,adj,edge):
        x2=x.view(48,-1)
        print(x2.shape)
        start=self.ggn(x2,edge) 
        print(start.shape)
        transG2=tnn.global_mean_pool(start,batch=torch.from_numpy(np.zeros((48))).long())
        print(transG2.shape)
        ## 48,200
        ### (54,48,200)
        forecast=[]

        self.ls.reset_hidden_state()
        for step in range(transG.shape[0]):
            
            h_t,c_t=self.ls(transG[step])    
            #lsG=self.linear2(h_t)   
            #lsG2.leaky_relu(lsG)
            ###self.ls.hidden_cell=lsG.clone()
            forecast.append(h_t)
        forecast2=torch.stack(forecast)
        for4=self.fc1(forecast2)
        for5=self.fc2(for4)                 
        for6=self.fc3(for5)                 ### first forecast
        #forecast3=torch.sigmoid(for)         
        return for6#torch.stack(outputs)#.squeeze_(0)#torch.transpose(out,0,1)#torch.stack(outputs)

    def gate(self,x,edge):
          x2=torch.transpose(x,0,1)
          x2=x2.reshape(48,-1)
          in2=self.ggn(x2,edge)
          return in2

    def testIt(self,x,adj,edge,predLength,Testdata):
        
        transG=[]
        for g in range(x.shape[0]):    ## x is [3,48,7]
            in2=self.gate(x,edge)        ## 48,3,7  ->48,21
            transG.append(in2)
        transG=torch.stack(transG)          ## 48,200
        

        ### (54,48,200)
        forecast=[]
        self.ls.reset_hidden_state()
        count=0
        
        
        tempo=x[0].clone()   ## 3,48,7
        for step in range(transG.shape[0]+predLength):
            count+=1
            
            #h_t,c_t=self.ls(transG[step]) 
            
            if step <transG.shape[0]:
                h_t,c_t=self.ls(transG[step])
                for4=self.fc1(h_t)
                for5=self.fc2(for4)
                for6=self.fc3(for5)
                forecast.append(for6)       ## First Forecast
                #print(for6.shape)
            else:

                combo=torch.cat((Testdata[step-transG.shape[0]][0].float(),for6.clone().unsqueeze_(1)),dim=1) ## test shape is (1,1,48,6)  ## h_t is # of cases in 
                tempo=torch.cat((tempo[-2:],combo.unsqueeze_(0)))  ## 3,48,7
                #print(tempo.shape)
                #print(combo.shape)
                #transG=torch.stack((transG,combo)) ## append pred to output List 
                #sample=transG[-3:]  ## Take last 3 samples 
                #print(tempo)
                embed=self.gate(tempo,edge) ## GGN
            
                h_t,c_t=self.ls(embed) ## LSTM Pred 
                for4=self.fc1(h_t)
                for5=self.fc2(for4)
                for6=self.fc3(for5)
                forecast.append(for6)
                #tempo+=[]
                #in2=self.ggn(h_t,edge)

        #print(forecast.shape)
        forecast=torch.stack(forecast)
        #forecast3=torch.sigmoid(for)
        return forecast#torch.stack(outputs)#.squeeze_(0)#torch.transpose(out,0,1)#torch.stack(outputs)



    def forward(self,x,adj,edge,predLength=None,Testdata=None,phase='Train'):
        if phase=='Train':
            cast=self.trainIt(x,adj,edge)
        else: 
            cast=self.testIt(x,adj,edge,predLength,Testdata)
        return cast#torch.stack(outputs)#.squeeze_(0)#torch.transpose(out,0,1)#torch.stack(outputs)


