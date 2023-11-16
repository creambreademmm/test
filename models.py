import torch
import numpy as np
import random

import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import init
import ipdb

#GCN layer
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):    
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output
            
# mymodel
class GCN(nn.Module):
    def __init__(self, nfeat, embed1, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, embed1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        return x

class MLP_1(nn.Module):
    def __init__(self, nfeat, nclass):
        super(MLP_1, self).__init__()
        self.mlp1 = nn.Linear(nfeat, nclass)

    def forward(self, x):
        x = self.mlp1(x) 
        return x

class MLP_2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP_2, self).__init__()
        self.mlp1 = nn.Linear(nfeat, nhid)
        self.mlp2 = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.mlp1(x))
        x = self.dropout(x)
        x = self.mlp2(x)
        return x

class MTL_1(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(MTL_1, self).__init__()
        self.mlp1 = nn.Linear(nfeat, nhid)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x =  F.relu(self.mlp1(x))
        x = self.dropout(x)
        return x

class MLP_3(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout):
        super(MLP_3, self).__init__()
        self.mlp1 = nn.Linear(nfeat, nhid1)
        self.mlp2 = nn.Linear(nhid1, nhid2)
        self.mlp3 = nn.Linear(nhid2, nclass)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.mlp1(x))
        x = self.dropout(x)
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)
        return x

class GCN_MLP_1(nn.Module):
    def __init__(self, nfeat, embed1, nclass, dropout):
        super(GCN_MLP_1, self).__init__()
        self.gc1 = GraphConvolution(nfeat, embed1)
        self.dropout = nn.Dropout(p=dropout)
        self.mlp1 = nn.Linear(embed1, nclass)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.mlp1(x)
        return x

class GCN_MLP_3(nn.Module):
    def __init__(self, nfeat, embed1, nhid1, nhid2, nclass, dropout):
        super(GCN_MLP_3, self).__init__()
        self.gc1 = GraphConvolution(nfeat, embed1)
        self.mlp1 = nn.Linear(embed1, nhid1)
        self.mlp2 = nn.Linear(nhid1, nhid2)
        self.mlp3 = nn.Linear(nhid2, nclass)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = F.relu(self.mlp1(x))
        x = self.dropout(x)
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)
        return x

class GCN_2_MLP_3(nn.Module):
    def __init__(self, nfeat, embed1, embed2, nhid1, nhid2, nclass, dropout):
        super(GCN_2_MLP_3, self).__init__()
        self.gc1 = GraphConvolution(nfeat, embed1)
        self.gc2 = GraphConvolution(embed1, embed2)
        self.mlp1 = nn.Linear(embed2, nhid1)
        self.mlp2 = nn.Linear(nhid1, nhid2)
        self.mlp3 = nn.Linear(nhid2, nclass)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = F.relu(self.gc2(x, adj))
        x = self.dropout(x)
        x = F.relu(self.mlp1(x))
        x = self.dropout(x)
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)
        return x

class MLP1(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid3, nclass, dropout):
        super(MLP1, self).__init__()
        self.mlp1 = nn.Linear(nfeat, nhid1)
        self.mlp2 = nn.Linear(nhid1, nhid2)
        self.mlp3 = nn.Linear(nhid2, nhid3)
        self.mlp4 = nn.Linear(nhid3, nclass)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.mlp1(x))
        x = self.dropout(x)
        x = F.relu(self.mlp2(x))
        x = self.dropout(x)
        x = F.relu(self.mlp3(x))
        x = self.mlp4(x)
        return x



