# -*- coding: utf-8 -*-


import torch
import tqdm
import sys, random
import argparse
import time, datetime
import os
import networkx as nx
import torch.nn as nn
from torch import Tensor as Tensor
from torch.nn import Linear as Linear
import torch.nn.init as init
from torch.nn.init import _calculate_correct_fan, calculate_gain
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, MessagePassing
import math
from torch_geometric import data
from torch.autograd import Variable as V
from torch_geometric.utils import add_self_loops, remove_self_loops, dense_to_sparse, add_remaining_self_loops
import numpy as np
import word2vec
from torch_geometric.data import DataLoader
from torch.utils.data.dataloader import default_collate
from helper import edges_mapping, cal_PMI, DataHelper

"""
### SGNN MODEL and other functions:###
"""
def pooling(x: torch.Tensor, batch_info, method):
    if method == 'add':
        return global_add_pool(x, batch_info['batch'], batch_info['num_graphs'])
    elif method == 'mean':
        return global_mean_pool(x, batch_info['batch'], batch_info['num_graphs'])
    elif method == 'max':
        return global_max_pool(x, batch_info['batch'], batch_info['num_graphs'])
    else:
        raise ValueError("Pooling method not implemented")


class BatchNorm(nn.Module):
    def __init__(self, channels: int, use_x: bool):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels)
        self.use_x = use_x

    def forward(self, u):
        if self.use_x:
            return self.bn(u)
        else:
            return self.bn(u.transpose(1, 2)).transpose(1, 2)


class EdgeCounter(MessagePassing):
    # def __init__(self):
    def __init__(self):
        super().__init__(aggr='max')

    def forward(self, x, edge_index, batch, batch_size):
        
        n_edges = self.propagate(edge_index, size=(x.size(0), x.size(0)), x = x)
        
        # return global_add_pool(n_edges, batch, batch_size)[batch]
        return global_mean_pool(n_edges, batch, batch_size)[batch]


class Linear(nn.Module):
    """ Linear layer with potentially smaller parameters at initialization. """
    __constants__ = ['bias', 'in_features', 'out_features']
    
    def __init__(self, in_features, out_features, bias=True, gain: float = 0.01):
        super().__init__()
        self.gain = gain
        self.lin = nn.Linear(in_features, out_features, bias)

    def reset_parameters(self):
  
        torch.nn.init.xavier_normal_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.normal_(self.lin.bias, 0, self.gain / math.sqrt(self.lin.out_features))

    def forward(self, x):
        return self.lin.forward(x)

class XtoGlobal(Linear):
    def forward(self, x: Tensor, batch_info: dict, method='mean'):
        """ x: (num_nodes, in_features). """
        g = pooling(x, batch_info, method)
        print(" x2global", g.size())  # bs, N, in_feat or bs, in_feat
        return self.lin.forward(g)

### each batch contains graph features for each graph sample  is this for extracting 
#### the topological property of graph

class UtoGlobal(nn.Module):
    def __init__(self, in_features: int , out_features: int, bias: bool, gain: float):
        super().__init__()
        self.lin1 = Linear(in_features, out_features, bias, gain=gain)
        self.lin2 = Linear(in_features, out_features, bias, gain=gain)

    def forward(self, u, batch_info: dict, method='mean'):
      #####------- int features is the word2vec embedding --------- #########
        """ u: (num_nodes, colors, in_features)
            output: (batch_size, out_features). """
      #######---------------------------------------###########
        coloring = batch_info['coloring']
        # Extract trace
        index_tensor = coloring[:, :, None].expand(u.shape[0], 1, u.shape[2])
        extended_diag = u.gather(1, index_tensor)[:, 0, :]          # n_nodes, in_feat
        mean_batch_trace = pooling(extended_diag, batch_info, 'mean')    # n_graphs, in_feat
        out1 = self.lin1(mean_batch_trace)                   # bs, out_feat
        # Extract sum of elements - trace
        mean = torch.sum (u/batch_info['n_batch'], dim=1)  # num_nodes, in_feat
        batch_sum = pooling(mean, batch_info, 'mean')                    # n_graphs, in_feat
        batch_sum = batch_sum - mean_batch_trace                         # make the basis orthogonal
        out2 = self.lin2(batch_sum)  # bs, out_feat
        return out1 + out2

class ChannelWiseU(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_groups=None):
        super().__init__()
        if n_groups is None:
            n_groups = in_features
        self.lin1 = torch.nn.Conv1d(in_features, out_features, kernel_size=1, groups=n_groups, bias=False)

    def forward(self, u):
        """ u: N x colors x channels. """
        u = u.transpose(1, 2)
        u = self.lin1(u)
        return u.transpose(1, 2)

#  
##first using the context info to compute the feature of the
## given nodes and once done, the other parts are masked again
## this part is for permutation equivariant processing 
## This is a schema-encoder
class SimpleUtoU(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool): #residual=False, n_groups=None):
        super().__init__()
        n_groups = 1 
        if n_groups is None:
            n_groups = out_features
        self.residual = False
        if self.residual is not True:
           print("residual is not set as defult of unapplied")

        self.lin1 = torch.nn.Conv1d(in_features, out_features, kernel_size=1, groups=n_groups, bias=True)
        self.lin2 = torch.nn.Conv1d(in_features, out_features, kernel_size=1, groups=n_groups, bias=True)
        self.lin3 = torch.nn.Conv1d(in_features, out_features, kernel_size=1, groups=n_groups, bias=True)

    def forward(self, u: Tensor, batch_info: dict = None):
        """ U: N x n_colors(the unique nodes are identified by the color) x channels"""
        old_u = u
        n = batch_info['num_nodes']
        num_colors = u.shape[1]
        out_feat = self.lin1.out_channels
        mask = batch_info['mask'][..., None].expand(n, num_colors, out_feat)
        normalizer = batch_info['n_batch']
        mean2 = torch.sum(u / normalizer, dim=1)     # N, in_feat
        mean2 = mean2.unsqueeze(-1)                  # N, in_feat, 1
        # 1. Transform u element-wise
        u = u.permute(0, 2, 1)
        out = self.lin1(u).permute(0, 2, 1)
        print("simpleu2u", out.size())

     # 2. Put in self of each line;sum over each line
        z2 = 0.1 * self.lin2(mean2)                            # N, out_feat, 1
        z2 = z2.transpose(1, 2)                          # N, 1, out_feat
        index_tensor = batch_info['coloring'][:, :, None].expand(out.shape[0], 1, out_feat)
        out.scatter_add_(1, index_tensor, z2)      # n, n_colors, out_feat the
                                                   #generate the indicator
                                                   
        # 3. Put everywhere the sum over each line
        z3 = 0.1 * self.lin3(mean2)                       # N, out_feat, 1
        z3 = z3.transpose(1, 2)                     # N, 1, out_feat
        out3 = z3.expand(n, num_colors, out_feat)
        out += out3 * mask                          # Mask the extra colors
        if self.residual:
            return old_u + out
        print("out simple u2u", out.size())
        return out

class Edge_C(MessagePassing):
    def __init__(self, in_channels, out_channels, x, edge_index, edge_w, batch, batch_size):
    # def __init__(self, data):
        super(Edge_C, self).__init__(aggr='add') #  "Max" aggregation.
       
        ######### edge_w is set for pre-train purpose ##########

        self.edge_w = edge_w
        self.act = torch.nn.ReLU()
        self.x = x
        self.edge_index = edge_index
        self.batch = batch
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(self):
        # x has shape [N, in_channels]
        n = self.x.size(0)
        return self.propagate(self.edge_index, size=(n,n), x=self.x) #
    
    def message(self, x_j):
        # x_j has shape [E, in_channels]
        print("x_j type $ size", x_j.type(),x_j.size())
        x_j = x_j.mul(self.edge_w)
        x_j = self.act(x_j)
        return x_j

# have U with d dimension using neural layers
class GraphExtractor(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_x: bool):
      
        super().__init__()
        self.use_x = use_x
        self.extractor = UtoGlobal(in_features, out_features,True, 0.01)
        self.lin = nn.Linear(out_features, out_features)

    def forward(self, u: Tensor, batch_info: dict):
        out = self.extractor(u, batch_info)
        out = out + self.lin(F.relu(out))
        return out

class TypeGNNLayer(MessagePassing):
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.use_x = False
        self.message_nn = SimpleUtoU(in_features, out_features, bias=True) 
        if self.use_x:
            self.alpha = nn.Parameter(torch.zeros(1, out_features), requires_grad=True).to(device)
        else:
            self.alpha = nn.Parameter(torch.zeros(1, 1, out_features), requires_grad=True).to(device)

    def forward(self, u, edge_index, batch_info):
        """ x corresponds either to node features or to the local context, depending on use_x."""
        # n = batch_info['num_nodes']
        n = u.size(0)
       
        if self.use_x and u.dim() == 1:
            u = u.unsqueeze(-1)
 
        print("mess warning the edge index size", edge_index.size(), edge_index.type())
        return self.propagate(edge_index, size=(n, n), u=self.message_nn(u, batch_info))
            
    def message(self, u_j: Tensor):
        print('u_j in typesgnn', u_j.type(), u_j.size())
        return u_j

    def update(self, aggr_u, u):
        aggr_u = aggr_u + self.alpha * u * aggr_u
        return aggr_u + u

def create_batch_info(x, edge_index, batch, batch_size, num_nodes, edge_counter):
    """ Compute some information about the batch that will be used by SGNN."""
    # x, batch, batch_size, num_nodes = data.x, data.batch, data.num_graphs, data.num_nodes

    # Compute some information about the batch
    # Count the number of nodes in each graph
    unique, n_per_graph = torch.unique(batch, return_counts=True)
    n_batch = torch.zeros_like(batch, dtype=torch.float)

    for value, n in zip(unique, n_per_graph):
        n_batch[batch == value] = n.float()

    # Count the average number of edges per graph
    dummy = x.new_ones((num_nodes, 1))
    print("dumm", dummy.size())
    print("edge_in", edge_index.size())
    # print("dummy", dummy.type())
    average_edges = edge_counter(dummy, edge_index, batch, batch_size)

    ###---------------------##

    coloring = x.new_zeros(num_nodes, dtype=torch.long)
    for i in range(batch_size):
        coloring[batch == i] = torch.arange(n_per_graph[i], device=x.device)
    coloring = coloring[:, None]
    n_colors = torch.max(coloring) + 1  # Indexing starts at 0

    mask = torch.zeros(num_nodes, n_colors, dtype=torch.bool, device=x.device)
    for value, n in zip(unique, n_per_graph):
        mask[batch == value, :n] = True

    # Aggregate into a dict
    batch_info = {'num_nodes': num_nodes,
                  'num_graphs': batch_size,
                  'batch': batch,
                  'n_per_graph': n_per_graph,
                  'n_batch': n_batch[:, None, None].float(),
                  'average_edges': average_edges[:, :, None],
                  'coloring': coloring,
                  'n_colors': n_colors,
                  'mask': mask}
    return batch_info


class SGNN(torch.nn.Module):
    def __init__(self, num_input_features: int, num_classes: int, num_layers: int, hidden: int,
                 hidden_final: int, dropout_prob: float, use_batch_norm: bool, use_x: bool, train:bool):
        super().__init__()
        self.use_x = use_x
        self.dropout_prob = dropout_prob
        self.use_batch_norm = use_batch_norm
        self.edge_counter = EdgeCounter()
        self.inn =  hidden  #num_input_features
        self.outt = hidden
        
        if not self.use_x:
            num_input_features = 1
        print("Use x:", use_x)

  ##### -----    this part is for setting the network architecture   -------- ######

        self.x2u = XtoGlobal(self.inn, hidden_final)

        self.ex_lin = nn.Linear(self.inn, hidden_final)

        self.e_bn = BatchNorm(self.inn, use_x=True)
  ##### -------------------------------------####

        self.no_prop = GraphExtractor(in_features=num_input_features, out_features=hidden_final, use_x=use_x)
        self.initial_lin = nn.Linear(num_input_features, hidden)
       
        self.convs = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        self.feature_extractors = torch.nn.ModuleList([])
        ### TypeGNNLayer
        for i in range(0, num_layers):
            self.convs.append(TypeGNNLayer(in_features=hidden, out_features=hidden))
            self.batch_norm_list.append(BatchNorm(hidden, use_x=False))
            self.feature_extractors.append(GraphExtractor(in_features=hidden, out_features=hidden_final, use_x=use_x))

        # Last layers
        self.after_conv = nn.Linear(hidden_final, hidden_final)
        self.final_lin = nn.Linear(hidden_final, num_classes)

    def forward(self, x, edge_index, batch, batch_size, edge_w, num_nodes):
        """ data.x: (num_nodes, num_features)"""

        print("-----------------------", edge_index.size())
        batch_info = create_batch_info(x, edge_index, batch, batch_size, num_nodes, edge_counter)

        # Create the context matrix
        u = x.new_zeros((num_nodes, batch_info['n_colors'])) #device=x.device)
        u.scatter_(1, batch_info['coloring'], 1)
        u = u[..., None]

        ###------------for residual connection----------------###
        
        emb_func = Edge_C(self.inn, self.outt, x, edge_index, edge_w, batch, batch_size)
        u_emb = emb_func.forward()
        print("u_emb", u_emb.size())
        u_emb = F.dropout(u_emb, p=self.dropout_prob)
        u_emb = torch.relu(u_emb)
        u_emb = self.e_bn(u_emb)
        emb_out = self.x2u(u_emb, batch_info, method="max" )
        print("emb graph batch", emb_out.size())

        ###-----------------------------------------------------###

        # Forward pass
        
        out = self.no_prop(u, batch_info)

        ##### ----------------------######

        u = self.initial_lin(u)
     
        for i, (conv, bn, extractor) in enumerate(zip(self.convs, self.batch_norm_list, self.feature_extractors)):
            if self.use_batch_norm and i > 0:
                u = bn(u)
            u = conv(u, edge_index, batch_info)
            global_features = extractor.forward(u, batch_info)
            print("conv loop", global_features.size())
            out += global_features / len(self.convs)


        # Two layer MLP with dropout and residual connections:
        ##### ----------------------######
        out = torch.relu(self.after_conv(out))+ emb_out # + out  ##out
        out = F.dropout(out, p=self.dropout_prob, training=self.training)
        out = self.final_lin(out)
        return out

