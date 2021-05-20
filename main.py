# -*- coding: utf-8 -*-
import numpy as np
import word2vec
import torch
import numpy as np
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
from torch_geometric.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.optim as optim

def get_loss(output, target):
    # if loss == "mse":
    return F.mse_loss(torch.absolute(output), target)



def test(loaders, model, is_validation=False):
    model.eval()
    correct = 0
    for data in loaders:
        with torch.no_grad():
            x, edge_index, batch, batch_size, edge_w, num_nodes = data.x, data.edge_index, data.batch, data.num_graphs, data.edge_attr, data.num_nodes
            x = V(x)
            x = x.cuda()
            # edge_index = V(edge_index, requires_grad = False)
            edge_index = edge_index.cuda()
            batch = V(batch)
            batch = batch.cuda()
            edge_w = V(edge_w, requires_grad = False)
            edge_w = edge_w.cuda()

            # pred = model(data.to(device))
            pred = model(x, edge_index, batch, batch_size, edge_w, num_nodes)
            ##loss is computed by mse
            pred_sq = torch.squeeze(pred,1)
            print(pred_sq.type(), data.y.type())
            los = get_loss(pred_sq[0].cpu(), data.y[0]) #get_loss(pred_sq[0], data.y[0])
            correct+=los

    total = len(loaders.dataset)
    return correct / total


def train(tr_dataset, te_dataset, writer, node_feature_size, class_num,\
        num_layers,  hidden_size,  final_hidden,  drop_out, tr_batch_size,batch_size,epochs, us_bn, use_x):
    
    loaders = DataLoader(tr_dataset, batch_size=tr_batch_size, shuffle=True)
    test_loader =  DataLoader(te_dataset, batch_size=batch_size, shuffle=False)

    model = TypeGNNLayer(num_input_features = node_feature_size, num_classes=class_num, num_layers = num_layers, hidden=hidden_size,
                 hidden_final=final_hidden, dropout_prob=drop_out, use_batch_norm=us_bn, use_x=use_x, train = True).to(device)
    # model.cuda()
    print(model)    
    
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(opt,
                                                step_size=50,
                                                gamma=0.92)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params", pytorch_total_params)
    
    #########------------------------------------######
    # train
    for epoch in range(epochs):
        total_loss = 0
        nan_counts = 0
        
        model.train()
        for data in loaders:
            x, edge_index, batch, batch_size, edge_w, num_nodes = data.x, data.edge_index, data.batch, data.num_graphs, data.edge_attr, data.num_nodes
           
            x = V(x, requires_grad = True)
            x = x.cuda()
            edge_index = V(edge_index, requires_grad = False)
            edge_index = edge_index.cuda()
            batch = V(batch)
            batch = batch.cuda() 
          
            edge_w = V(edge_w, requires_grad = False)
            edge_w = edge_w.cuda()
    
            pred = model(x, edge_index, batch, batch_size, edge_w, num_nodes)

            pred_sq = torch.squeeze(pred,1)
           
            loss_mse = get_loss(pred_sq[0].cpu(), data.y[0])
   
            loss_mse.backward()  #retain_graph=True

            torch.nn.utils.clip_grad_norm_(model.parameters(),1)

            opt.step()
            opt.zero_grad()


            total_loss += loss_mse/len(data)

        total_loss /= len(loaders)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            print("Epoch {}. mse Loss: {:.4f}. mse Test: {:.4f}".format(
                epoch, total_loss, test_acc))

    return model

# from tensorboardX import SummaryWriter

writer = None  #SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))


rand = rand
max_length= max_l
SEED = rand

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

node_feature_size = node_f_size
class_num = class_num
epochs = epochs
num_layers = no_layer
hidden_size = h_size
final_hidden = f_size
batch_size = batch_size
tr_batch_size = tr_batch_size
drop_out = drop_out
us_bn = True
use_x = True
tr_data_list=tr_dataset
t_datalist = t_dataset


model = train(tr_data_list, t_datalist, writer, node_feature_size, class_num,\
        num_layers,  hidden_size,  final_hidden,  drop_out, tr_batch_size,batch_size,epochs, us_bn, use_x)


