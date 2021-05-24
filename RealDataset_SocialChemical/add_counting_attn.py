#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import SAGEConv, global_max_pool
from torch_geometric.utils import to_undirected, to_dense_adj, to_networkx, to_scipy_sparse_matrix

import torch_geometric
from deepwalk import OnlyWalk
import networkx as nx
from tqdm import tqdm



name = 'IMDB-BINARY'

path_length = 4
num_random_walk = 50
p=1
q=1
stopping_prob = 0

project_path = '' # set the project path for data here


#for path_length in [3,4,5]:

dataset = torch.load( project_path + name+'.pt')

for dt in tqdm(dataset):
    #break

    adj_mat = to_dense_adj(dt.edge_index)[0]
    if adj_mat.shape[0] != dt.x.shape[0]:
        extra_shape = dt.x.shape[0] - adj_mat.shape[0]
        adj_mat1 = torch.cat((adj_mat, torch.zeros(extra_shape, adj_mat.shape[0])), dim = 0)
        adj_mat = torch.cat((adj_mat1, torch.zeros(adj_mat1.shape[0], extra_shape)), dim = 1)


    graph = nx.to_networkx_graph(adj_mat.numpy(), create_using=nx.DiGraph)
    n2v = OnlyWalk.Node2vec_onlywalk(graph = graph, path_length=path_length, num_paths=num_random_walk, p=p, q=q, stop_prob = stopping_prob, with_freq_mat = True)
    counting_atten = torch.from_numpy(n2v.walker.freq_mat)
    counting_attn = counting_atten / (np.diag(counting_atten)[:, None])
    
    dt['counting_attn'] = {'mat':counting_attn,
                           'settings': "path_length={}, num_rw={}, p={}, q={}, stop_prob={}".format(path_length, num_random_walk, p, q, stopping_prob)}
    


torch.save(dataset, project_path + name + '_pathlen{}.pt'.format(path_length))


