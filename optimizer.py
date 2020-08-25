import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def mask(graph):
    c1=graph
    c11=(c1!=0).sum(axis=1)
    c12=c11[c11!=0]
    
    #c13=torch.zeros((c12.shape[0],c12.shape[1]))
    #c13[:len(c12)]=1
    return len(c12)


def tensMask(tens):
    t1=[]
    for i in range(tens.shape[0]):
        t1.append(mask(tens[i]))
    return t1



def cutLoss(r1,labels):
    inds=tensMask(labels)
    loss=0
    for k in range(len(inds)):
        #loss+= F.mse_loss(r1[k,:inds[k]], labels[k,:inds[k],[0,2]])/inds[k]#, pos_weight=pos_weight)
        loss+=((r1[k,:inds[k],:]-labels[k,:inds[k],[0,1,2]])**2).sum()/(inds[k])
    #print(inds[k])
    #print(r1.shape[0])
    return torch.sqrt(loss/(r1.shape[0]))




#test=torch.stack(tensMask(c3))


def loss_function(r1,labels, mu, logvar, n_nodes):
    #cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    cost1=cutLoss(r1,labels)
   # cost1 =   F.mse_loss(r1, labels[:,:,[0,2]])#, pos_weight=pos_weight)
   
    #, pos_weight=pos_weight)


    #cost2 =   torch.nn.BCELoss(r1, labels[:,:,[0,2]])#, pos_weight=pos_weight)

    
    #cost2= torch.nn.BCEWithLogitsLoss(r2, labels[:,:,1])
    #F.mse_loss(Xreco[:, n, :, :], X[:, n, :, :])
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 *(torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)))
    return cost1  + KLD 
