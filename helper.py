# -*- coding: utf-8 -*-

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
import numpy as np
import word2vec
from torch_geometric.data import DataLoader
from torch.utils.data.dataloader import default_collate


"""## add window size create connect matrix ##"""

def edges_mapping(vocab_len, content, ngram):
  count = 1
  mapping = np.zeros(shape = (vocab_len, vocab_len), dtype = np.int32)
  for doc in content:
    for i , src in enumerate(doc):
      for dst_id in range(max(0, i-ngram), min(len(doc), i+ngram+1)):
        dst = doc(dst_id)
        if mapping[src, dst] == 0:
          mapping[src, dst] = count
          count +=1
#add self node connection
  for word in range(vocab_len):
    mapping[word, word] = count
    count += 1
  return count, mapping

from sklearn.metrics import accuracy_score,mean_squared_error

def get_time_dif(start_time):
  end_time = time.time()
  time_dif = end_time - start_time
  return datetime.timedelta(seconds = int(round(time_dif)))

"""pmi"""

def cal_PMI(window_size=15, mode='train'):  
    helper = DataHelper(mode)
    content, _ = helper.get_content()
    pair_count_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)
    word_count =np.zeros(len(helper.vocab), dtype=int)
    
    for sentence in content:
        sentence = sentence.split(' ')
        for i, word in enumerate(sentence):
            try:
                word_count[helper.d[word]] += 1
            except KeyError:
                continue
            start_index = max(0, i - window_size)
            end_index = min(len(sentence), i + window_size)
            for j in range(start_index, end_index):
                if i == j:
                    continue
                else:
                    target_word = sentence[j]
                    try:
                        pair_count_matrix[helper.d[word], helper.d[target_word]] += 1
                    except KeyError:
                        continue
        
    total_count = np.sum(word_count)
    word_count = word_count / total_count
    pair_count_matrix = pair_count_matrix / total_count
    # print(pair_count_matrix)
    pmi_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=float)
    for i in range(len(helper.vocab)):
        for j in range(len(helper.vocab)):
            pmi_matrix[i, j] = np.log(
                pair_count_matrix[i, j] / (word_count[i] * word_count[j]) 
            )
            if pmi_matrix[i, j] <= 0:
              continue

    pmi_matrix = np.nan_to_num(pmi_matrix)
    
    pmi_matrix = np.maximum(pmi_matrix, 0.0)

    edges_weights = [0.0]
    count = 1
    edges_mappings = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)
    for i in range(len(helper.vocab)):
        for j in range(len(helper.vocab)):
            if pmi_matrix[i, j] != 0:
                edges_weights.append(pmi_matrix[i, j])
                edges_mappings[i, j] = count
                count += 1

    edges_weights = np.array(edges_weights)

    edges_weights = edges_weights.reshape(-1, 1)
    print("edges_weights shape", edges_weights.shape)
    edges_weights = torch.Tensor(edges_weights)
    
    return edges_weights, edges_mappings, count



"""##DATA Helper##

### This part is for seting the vocab_5 with no repetition ###
"""

class DataHelper(object):
    def __init__(self, mode='train', vocab=None):
        

        self.mode = mode

        self.base = os.path.join('data')

        self.current_set = os.path.join(self.base, '%s-stemmed.txt' % (self.mode))
        self.labels_str = 1

        content, label = self.get_content()
        

        if vocab is None:
            self.vocab = []

            try:
                self.get_vocab()
            except FileNotFoundError:
                self.build_vocab(content, min_count=5)
        else:
            self.vocab = vocab

        self.d = dict(zip(self.vocab, range(len(self.vocab))))
      
        self.content = [list(map(lambda x: self.word2id(x), doc.split(' '))) for doc in content]

        self.label =  self.label_to_array(label)
        
        
    def label_to_array(self, label):
        num = []
        for l in label:
          if l != '':
            num.append(int(l))

        return np.array(num)

    def get_content(self):
        with open(self.current_set) as f:
            all = f.read()
            content = [line.split('\t') for line in all.split('\n')]

            cleaned_l = []
            cleaned_c = []

            for pair in (content):
                l_abel = pair[0][0:2]
                
                c_ontent = pair[0][3:]
                if c_ontent == '' or l_abel == '':
                    pass
                
                cleaned_c.append(c_ontent)
                cleaned_l.append(l_abel)

        label, content = zip([cleaned_l, cleaned_c])
        return content[0], label[0]
        

    def word2id(self, word):
        try:
            result = self.d[word]
        except KeyError:
            result = self.d['UNK']

        return result

    def get_content_each(self):
        with open(self.current_set) as f:
            all = f.read()
            content = [line.split('\t') for line in all.split('\n')]

            cleaned_l = []
            cleaned_c = []

            for pair in (content):
                l_abel = pair[0][0:2]
                c_ontent = pair[0][3:]
                if c_ontent == '' or l_abel == '':
                    pass
                
                
                cleaned_c.append(c_ontent)
                cleaned_l.append(l_abel)
      

        for c in  cleaned_c:
            words = c.split(' ')
            for word in words:
                if word not in self.vocab:
                    self.vocab.append(word)

        return self.vocab   

    def get_vocab(self):
        with open(os.path.join(self.base, 'vocab_5.txt')) as f:
            vocab = f.read()
            self.vocab= vocab.split('\n')

    
   
    def build_vocab(self, content, min_count=10):
        vocab = []
        freq = dict(zip(vocab, [0 for i in range(len(vocab))]))

        for c in content:
          words = c.split(' ')
          for word in words:

            freq[word] +=1

        results = []
        for word in freq.keys():
          if freq[word] < min_count:
            continue
          else:
            results.append(word)

        results.insert(0, 'UNK')
        with open(os.path.join(self.base, 'vocab_5.txt'),'w') as f:
          f.write('\n'.join(results))
        self.vocab = results


        for c in content:
            words = c.split(' ')
            for word in words:
                if word not in vocab:
                    vocab.append(word)

        freq = dict(zip(vocab, [0 for i in range(len(vocab))]))

        for c in content:
            words = c.split(' ')
            for word in words:
                freq[word] += 1

        results = []
        for word in freq.keys():
            if freq[word] < min_count:
                continue
            else:
                results.append(word)

        results.insert(0, 'UNK')
        with open(os.path.join(self.base, 'vocab_5.txt'), 'w') as f:
            f.write('\n'.join(results))
        # print("the orginal vocab_5 len:{:2d}the current len is:{:2d}".format(len(results),len(self.vocab)))

    def batch_iter(self, batch_size, num_epoch):
        for i in range(num_epoch):
            num_per_epoch = int(len(self.content) / batch_size)
            print("content size", len(self.content), num_per_epoch)
            for batch_id in range(num_per_epoch):
                start = batch_id * batch_size
                end = min((batch_id + 1) * batch_size, len(self.content))

                content = self.content[start:end]
                label = self.label[start:end]

                print("content in num_epoch", len(content[0]))

                # yield content, torch.tensor(label).cuda(), i
                yield content, label, i


def graph_data(data_helper):
    
    pmi, edges_matrix, edg_nums = cal_PMI(window_size=15, mode='train')
    index = 0
    data_list = []
    seq_edge_w = torch.zeros((edg_nums,1))

    for content, label , _ in data_helper.batch_iter(batch_size = 1, num_epoch=1):
        
        vocab_c = data_helper.vocab
        
        print("content lenth each data", len(vocab_c))

        ###----------this is for the original operating mode  --------#########
        
        f = feature_etr(vocab_c)

        ##### --------- ---------------  #######
        print('file no', index)
        index += 1

        e,n,_, edg_ar = graphcon(content,label, edges_matrix, pmi)
        
        ##--------------------------------------------------##
        edges1 = [np.array([edge[0], edge[1]]) for edge in e]
        edge_index = torch.tensor(np.array(edges1).T, dtype=torch.long) #.cuda()
        edge_attr = torch.tensor(seq_edge_w[edg_ar], dtype = torch.float)   #.cuda()
      
        # print("edge atr", edge_attr.size())
        # edge_index, _ = add_remaining_self_loops (edge_index, edge_attr)
        print("edge index size", edge_index.size())
        ####-------------------------------###

        ft = torch.tensor(f, dtype=torch.float) #.cuda()
        y  = torch.tensor(label, dtype= torch.float)
        data_list.append(data.Data(x=ft, edge_index= edge_index, edge_attr=edge_attr, y=y))
    
    return  data_list

# for d in data_list:
n_samples = len(data_list)
torch.save(data_list,'data'+ f'/dep_{n_samples}_train.pt')



def to_torch_geom(adj, features, graph_labels, debug = True):
    graphs = []
    for i in range(len(adj)):          # Graph of a given size
        print("len adj", len(adj))
        batch_i = []
        for j in range(adj[i].shape[0]):       # Number of graphs
            graph_adj = adj[i][j] ## [edge_index, edge_attribute]
            graph = data.Data( x = features[i][j],
                              edge_index = (graph_adj)[0],
                              y=graph_labels[i][j].unsqueeze(0))
                              # , pos=node_labels[i][j])
            if not debug:
                batch_i.append(graph)
        if debug:
            batch_i.append(graph)
        graphs.append(batch_i)
    return graphs.to(device)


