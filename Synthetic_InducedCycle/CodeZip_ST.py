#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import networkx as nx
import matplotlib.pyplot as plt 
import time
import numpy as np
import pickle
from tqdm.notebook import tqdm, trange
import random

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.data import MiniGCDataset
from dgl.nn.pytorch import *
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm, trange

import seaborn as sns

from random import shuffle
from multiprocessing import Pool
import multiprocessing
from functools import partial
from networkx.generators.classic import cycle_graph
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from deepwalk import OnlyWalk

import os, sys






class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#### Graph Generations

def shuffle_two_lists(list_1, list_2):

    c = list(zip(list_1, list_2))
    random.shuffle(c) 
    return zip(*c)




#%%
##### Spanning Tree Synthetic Graph


from collections import deque

# method return farthest node and its distance from node u
def BFS(graph_in, u):
    # marking all nodes as unvisited
    visited = [False for i in range(len(graph_in.nodes) + 1)]
    # mark all distance with -1
    distance = [-1 for i in range(len(graph_in.nodes)  + 1)]

    # distance of u from u will be 0
    distance[u] = 0
    # in-built library for queue which performs fast oprations on both the ends
    queue = deque()
    queue.append(u)
    # mark node u as visited
    visited[u] = True

    while queue:

        # pop the front of the queue(0th element)
        front = queue.popleft()
        # loop for all adjacent nodes of node front

        for i in [x for x in graph_in.neighbors(front)]:
            if not visited[i]:
                # mark the ith node as visited
                visited[i] = True
                # make distance of i , one more than distance of front
                distance[i] = distance[front]+1
                # Push node into the stack only if it is not visited already
                queue.append(i)

    maxDis = 0

    # get farthest node distance and its index
    for i in range(len(graph_in.nodes) ):
        if distance[i] > maxDis:

            maxDis = distance[i]
            nodeIdx = i

    return nodeIdx, maxDis


def FindLongestNodePairs(graph_in):
    # first DFS to find one end point of longest path
    node, Dis = BFS(graph_in, 0)

    # second DFS to find the actual longest path
    node_2, LongDis = BFS(graph_in, node)

    return node, node_2




#%%

def generate_tree(num_nodes, num_graphs):

    all_tree_graphs = []
    print(f'Start generating tree graphs')
    for i in trange(num_graphs):
      all_tree_graphs.append(nx.generators.trees.random_tree(num_nodes))
    return all_tree_graphs

def generate_tree_adding_edges(num_nodes, num_graphs, num_edges = 3):

    all_tree_graphs_adding_edges = []
    print(f'Start generating tree graphs with edges')
    for i in trange(num_graphs):
      tree = nx.generators.trees.random_tree(num_nodes)
      for j in range(num_edges):
        while True:
          vertices = random.sample(range(num_nodes), 2)
          if ((vertices[0], vertices[1]) not in tree.edges) and ((vertices[1], vertices[0]) not in tree.edges):
            tree.add_edge(vertices[0], vertices[1])
            break
          else:
            continue
      all_tree_graphs_adding_edges.append(tree)
    return all_tree_graphs_adding_edges


def generate_tree_adding_edges_with_longest_distance(num_nodes, num_graphs, num_edges = 3):
    
    #num_edges = 1

    all_tree_graphs_adding_edges = []
    print(f'Start generating tree graphs with edges with longest distance')
    for i in trange(num_graphs):
      tree = nx.generators.trees.random_tree(num_nodes)
      for j in range(num_edges):
        #while True:
        vertices = FindLongestNodePairs(tree)
        #vertices = random.sample(range(num_nodes), 2)
        #if ((vertices[0], vertices[1]) not in tree.edges) and ((vertices[1], vertices[0]) not in tree.edges):
        tree.add_edge(vertices[0], vertices[1])
        #  break
        #else:
        #  continue
      all_tree_graphs_adding_edges.append(tree)
    return all_tree_graphs_adding_edges


def generate_tree_adding_edges_with_shortest_distance(num_nodes, num_graphs, num_edges = 3):
    
    #num_edges = 1

    all_tree_graphs_adding_edges = []
    print(f'Start generating tree graphs with edges with longest distance')
    for i in trange(num_graphs):
      while True:
          tree = nx.generators.trees.random_tree(num_nodes)
          vertices = FindLongestNodePairs(tree)
          longest_len = nx.shortest_path_length(tree,source=vertices[0],target=vertices[1])

          source = np.random.choice(len(tree.nodes))
          target = np.random.choice(len(tree.nodes))
          if source != target:
              if nx.shortest_path_length(tree,source=source,target=target) >= 2:
                  if  nx.shortest_path_length(tree,source=source,target=target) < longest_len:
                      tree.add_edge(source, target)
                      all_tree_graphs_adding_edges.append(tree)
                      break
      #for j in range(num_edges):
      #  source = np.random.choice(len(tree.nodes))
      #  for target in tree.nodes:
      #    if nx.shortest_path_length(tree,source=source,target=target) == 2:
      #      tree.add_edge(source, target)
      #      break
      #all_tree_graphs_adding_edges.append(tree)
    return all_tree_graphs_adding_edges

    

#%%

def generate_graphs_labels(num_nodes, num_train_tree, num_train_edge_tree,
                           num_val_tree, num_val_edge_tree, num_test_tree,
                           num_test_edge_tree, num_edges, is_dgl_type = False,
                           path_length = 10, num_random_walk= 50, p=1e3, q=1, stopping_prob = 0.0):

    tree_train_graphs = generate_tree_adding_edges_with_shortest_distance(num_nodes, num_train_tree, num_edges)
    tree_train_labels = list(np.zeros(num_train_tree))
    edge_tree_train_graphs = generate_tree_adding_edges_with_longest_distance(num_nodes, num_train_edge_tree, num_edges)
    edge_tree_train_labels = list(np.ones(num_train_edge_tree))

    tree_val_graphs = generate_tree_adding_edges_with_shortest_distance(num_nodes, num_val_tree, num_edges)
    tree_val_labels = list(np.zeros(num_val_tree))
    edge_tree_val_graphs = generate_tree_adding_edges_with_longest_distance(num_nodes, num_val_edge_tree, num_edges)
    edge_tree_val_labels = list(np.ones(num_val_edge_tree))

    tree_test_graphs = generate_tree_adding_edges_with_shortest_distance(num_nodes, num_test_tree, num_edges)
    tree_test_labels = list(np.zeros(num_test_tree))
    edge_tree_test_graphs = generate_tree_adding_edges_with_longest_distance(num_nodes, num_test_edge_tree, num_edges)
    edge_tree_test_labels = list(np.ones(num_test_edge_tree))

    all_train_graphs = tree_train_graphs + edge_tree_train_graphs
    all_train_labels = tree_train_labels + edge_tree_train_labels

    all_val_graphs = tree_val_graphs + edge_tree_val_graphs
    all_val_labels = tree_val_labels + edge_tree_val_labels

    all_test_graphs = tree_test_graphs + edge_tree_test_graphs
    all_test_labels = tree_test_labels + edge_tree_test_labels

    all_train_graphs_shuffled, all_train_labels_shuffled = \
                          shuffle_two_lists(all_train_graphs, all_train_labels)
    all_val_graphs_shuffled, all_val_labels_shuffled = \
                          shuffle_two_lists(all_val_graphs, all_val_labels)
    all_test_graphs_shuffled, all_test_labels_shuffled = \
                         shuffle_two_lists(all_test_graphs, all_test_labels)
    
    all_train_graphs_shuffled = list(all_train_graphs_shuffled)
    all_train_labels_shuffled = list(all_train_labels_shuffled)
    all_val_graphs_shuffled = list(all_val_graphs_shuffled)
    all_val_labels_shuffled = list(all_val_labels_shuffled)
    all_test_graphs_shuffled = list(all_test_graphs_shuffled)
    all_test_labels_shuffled = list(all_test_labels_shuffled)


    return all_train_graphs_shuffled, all_train_labels_shuffled,\
           all_val_graphs_shuffled, all_val_labels_shuffled,\
           all_test_graphs_shuffled, all_test_labels_shuffled



def networkx_to_dgl_graphs(all_train_graphs_shuffled, all_val_graphs_shuffled, all_test_graphs_shuffled):

    for i in range(len(all_train_graphs_shuffled)):
        all_train_graphs_shuffled[i] = dgl.from_networkx(all_train_graphs_shuffled[i])

    for i in range(len(all_val_graphs_shuffled)):
        all_val_graphs_shuffled[i] = dgl.from_networkx(all_val_graphs_shuffled[i]) 

    for i in range(len(all_test_graphs_shuffled)):
        all_test_graphs_shuffled[i] = dgl.from_networkx(all_test_graphs_shuffled[i])

    return all_train_graphs_shuffled, all_val_graphs_shuffled, all_test_graphs_shuffled
    


def dgl_to_networkx_graphs(all_train_graphs_shuffled, all_val_graphs_shuffled, all_test_graphs_shuffled):

    for i in range(len(all_train_graphs_shuffled)):
        all_train_graphs_shuffled[i] = nx.Graph(all_train_graphs_shuffled[i].to_networkx())

    for i in range(len(all_val_graphs_shuffled)):
        all_val_graphs_shuffled[i] = nx.Graph(all_val_graphs_shuffled[i].to_networkx())

    for i in range(len(all_test_graphs_shuffled)):
        all_test_graphs_shuffled[i] = nx.Graph(all_test_graphs_shuffled[i].to_networkx())

    return all_train_graphs_shuffled, all_val_graphs_shuffled, all_test_graphs_shuffled
    

    #GWK_masking = generate_masking_GWK(all_train_graphs_shuffled, all_val_graphs_shuffled, all_test_graphs_shuffled, path_length, num_random_walk, stopping_prob, p, q)
    #GAT_masking = generate_masking_GAT(all_train_graphs_shuffled, all_val_graphs_shuffled, all_test_graphs_shuffled)
    




#%%
##### Generate masking

def generate_masking_GAT(train_graphs, val_graphs, test_graphs):

  train_masking = []
  val_masking = []
  test_masking = []

  print('Start generating GAT masking')

  for graph in train_graphs:
      adj = nx.linalg.graphmatrix.adjacency_matrix(graph).todense()
      np.fill_diagonal(adj, 1)
      train_masking.append(torch.from_numpy(adj))
  
  for graph in val_graphs:
      adj = nx.linalg.graphmatrix.adjacency_matrix(graph).todense()
      np.fill_diagonal(adj, 1)
      val_masking.append(torch.from_numpy(adj))

  for graph in test_graphs:
      adj = nx.linalg.graphmatrix.adjacency_matrix(graph).todense()
      np.fill_diagonal(adj, 1)
      test_masking.append(torch.from_numpy(adj))

  return train_masking, val_masking, test_masking




def generate_masking_GWK(train_graphs, val_graphs, test_graphs, num_random_walk, path_length, stopping_prob, p, q, ignore_start = False):

  train_masking = []
  val_masking = []
  test_masking = []

  print('Start generating GWK masking')

  for i in tqdm(range(len(train_graphs))):
      
      graph = (train_graphs[i])
      n2v = OnlyWalk.Node2vec_onlywalk(graph = graph, path_length=path_length, num_paths=num_random_walk, p=p, q=q, stop_prob = stopping_prob, with_freq_mat = True)
      counting_atten = torch.from_numpy(n2v.walker.freq_mat)
      if ignore_start:
          counting_atten -= np.eye(len(counting_atten))*num_random_walk
      train_masking.append(MinMaxScaler(counting_atten).float())

  for i in tqdm(range(len(val_graphs))):
      
      graph = (val_graphs[i])
      n2v = OnlyWalk.Node2vec_onlywalk(graph = graph, path_length=path_length, num_paths=num_random_walk, p=p, q=q, stop_prob = stopping_prob, with_freq_mat = True)
      counting_atten = torch.from_numpy(n2v.walker.freq_mat)
      if ignore_start:
          counting_atten -= np.eye(len(counting_atten))*num_random_walk
      val_masking.append(MinMaxScaler(counting_atten).float())

  for i in tqdm(range(len(test_graphs))):

      graph = (test_graphs[i])
      n2v = OnlyWalk.Node2vec_onlywalk(graph = graph, path_length=path_length, num_paths=num_random_walk, p=p, q=q, stop_prob = stopping_prob, with_freq_mat = True)
      counting_atten = torch.from_numpy(n2v.walker.freq_mat)
      if ignore_start:
          counting_atten -= np.eye(len(counting_atten))*num_random_walk
      test_masking.append(MinMaxScaler(counting_atten).float())

  return train_masking, val_masking, test_masking





#%%

##### Better Version

def counting_attn(node, epsilon, adj_mat, discount_factor, rand_seed=None):

  if rand_seed:
    np.random.seed(rand_seed)
  else:
    np.random.seed()

  counting_vector = np.zeros(adj_mat.shape[1])
  counting_vector[node] = 1
  step_length = 0 
  visited_nodes = [node]
  loop_count = 0
  while True:

    continue_or_not = np.random.choice([0, 1], p = [epsilon, 1 - epsilon])

    if continue_or_not:
      while True:
        next_available_nodes = np.where(adj_mat[node, :] != 0)[0]

        if len(set(next_available_nodes) - set(visited_nodes))>0:

          next = np.random.choice(list(set(next_available_nodes) - set(visited_nodes)))
          step_length += 1
          counting_vector[next] += discount_factor**step_length
          visited_nodes.append(next)
          node = next
        else:
          return counting_vector
        '''
        if next in visited_nodes:
          loop_count +=1
          if loop_count > 1000:
            #print('exceed loop count')
            return counting_vector
          else:
            continue
        else:
        '''


    else:
      return counting_vector

def cal_counting_attn(adj, num_random_walks, stopping_prob, discounting_fact,
                      seed = 666):

    np.random.seed(seed)
    nb_nodes = adj.shape[0]

    #counting_attn
    vector_dict = []
    for node in range(nb_nodes):
      all_vectors = []
      for i in range(num_random_walks):
        try:
            vector = counting_attn(node, stopping_prob, np.array(adj),
                                   discounting_fact, i+1)
        except:
            vector = np.zeros(np.array(adj).shape[1])
            vector[node] = 1
        all_vectors.append(vector)
      vector_dict.append(all_vectors)
    return vector_dict 


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def MinMaxScaler(data):
    diff = data.transpose(0,1) - torch.min(data, axis = 1)[0]
    range = torch.max(data, axis = 1)[0] - torch.min(data, axis = 1)[0]
    return (diff / (range + 1e-7)).transpose(0,1)





#%%
# Graph NN

##### Attention Model definition

class GWKLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 feat_drop=0.,
                 attn_drop=0.,
                 alpha=0.2,
                 agg_activation=F.elu):
        super(GWKLayer, self).__init__()

        self.feat_drop = nn.Dropout(feat_drop)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        #torch.nn.init.zeros_(self.fc.bias)
        self.attn_l = nn.Parameter(torch.ones(size=(out_dim, 1)))
        self.attn_r = nn.Parameter(torch.ones(size=(out_dim, 1)))
        self.attn_drop = nn.Dropout(attn_drop)
        self.activation = nn.LeakyReLU(alpha)
        self.softmax = nn.Softmax(dim = 1)

        self.agg_activation=agg_activation

    def clean_data(self):
        ndata_names = ['ft', 'a1', 'a2']
        edata_names = ['a_drop']
        for name in ndata_names:
            self.g.ndata.pop(name)
        for name in edata_names:
            self.g.edata.pop(name)

    def forward(self, feat, bg, counting_attn):
      #with HiddenPrints():
        # prepare, inputs are of shape V x F, V the number of nodes, F the dim of input features
        self.g = bg
        h = self.feat_drop(feat)
        #print('h shape is \n')
        #print(h.shape)
        head_ft = self.fc(h).reshape((h.shape[0], -1))
        #print('ft shape is \n')
        #print(head_ft.shape)
        
        a1 = torch.mm(head_ft, self.attn_l)    # V x 1
        a2 = torch.mm(head_ft, self.attn_r)     # V x 1
        a = self.attn_drop(a1 + a2.transpose(0, 1))
        a = self.activation(a)
        #print('a shape is \n')
        #print(a.shape)

        #maxes = torch.max(a, 1, keepdim=True)[0]
        a_ = a #- maxes
        a_nomi = torch.mul(torch.exp(a_), counting_attn.float())
        a_deno = torch.sum(a_nomi, 1, keepdim=True)
        a_nor = a_nomi/(a_deno+1e-9)

        ret = torch.mm(a_nor, head_ft)
        #print('ret shape is \n')
        #print(ret.shape)
        if self.agg_activation is not None:
            ret = self.agg_activation(ret)

        return ret





class GWKClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, feat_drop_=0.,
                 attn_drop_=0.,):
        super(GWKClassifier, self).__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            nn.ModuleList([GWKLayer(in_dim, hidden_dim, feat_drop = feat_drop_,
                                    attn_drop = attn_drop_, agg_activation=F.elu)
            for _ in range(num_heads)]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads, hidden_dim,
                                    feat_drop = feat_drop_, attn_drop = attn_drop_,
                                    agg_activation=F.elu)
            for _ in range(1)]),
        ])
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, bg, counting_attn, normalize = 'normal'):
        # For undirected graphs, in_degree is the same as
        # out_degree.

        h = bg.in_degrees().view(-1, 1).float()
        #print('input shape is \n')
        #print(h.shape)
        num_nodes = h.shape[0]
        
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)
        #if std_:
        #   h = (h - mean_)/std_
        #else:   
        #   h = h/np.max(features)

        for i, gnn in enumerate(self.layers):
            all_h = []
            for j, att_head in enumerate(gnn):
                all_h.append(att_head(h, bg, counting_attn))   
            h = torch.squeeze(torch.cat(all_h, dim=1))
            #print('Output shape is \n')
            #print(h.shape)

        bg.ndata['h'] = h
        hg = dgl.mean_nodes(bg, 'h')
        #return self.softmax(self.classify(hg))
        return self.classify(hg)






class GWKClassifier_2hid(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, feat_drop_=0.,
                 attn_drop_=0.,):
        super(GWKClassifier_2hid, self).__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            nn.ModuleList([GWKLayer(in_dim, hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads)]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads, hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(1)])
        ])
        self.classify = nn.Linear(hidden_dim * 1, n_classes)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, bg, counting_attn, normalize = 'normal'):

        h = bg.in_degrees().view(-1, 1).float()
        num_nodes = h.shape[0]
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)

        for i, gnn in enumerate(self.layers):
            all_h = []
            for j, att_head in enumerate(gnn):
                all_h.append(att_head(h, bg, counting_attn))   
            h = torch.squeeze(torch.cat(all_h, dim=1))

        bg.ndata['h'] = h
        hg = dgl.mean_nodes(bg, 'h')
        return self.classify(hg)




class GWKClassifier_3hid(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, feat_drop_=0.,
                 attn_drop_=0.,):
        super(GWKClassifier_3hid, self).__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            nn.ModuleList([GWKLayer(in_dim, hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[0])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[0], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[1])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[1], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(1)])
        
        ])
        self.classify = nn.Linear(hidden_dim , n_classes)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, bg, counting_attn, normalize = 'normal'):

        h = bg.in_degrees().view(-1, 1).float()
        num_nodes = h.shape[0]
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)

        for i, gnn in enumerate(self.layers):
            all_h = []
            for j, att_head in enumerate(gnn):
                all_h.append(att_head(h, bg, counting_attn))   
            h = torch.squeeze(torch.cat(all_h, dim=1))

        bg.ndata['h'] = h
        hg = dgl.mean_nodes(bg, 'h')
        return self.classify(hg)



class GWKClassifier_4hid(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, feat_drop_=0.,
                 attn_drop_=0.,):
        super(GWKClassifier_4hid, self).__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            nn.ModuleList([GWKLayer(in_dim, hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[0])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[0], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[1])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[1], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[2])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[2], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(1)])
        ])
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, bg, counting_attn, normalize = 'normal'):

        h = bg.in_degrees().view(-1, 1).float()
        num_nodes = h.shape[0]
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)

        for i, gnn in enumerate(self.layers):
            all_h = []
            for j, att_head in enumerate(gnn):
                all_h.append(att_head(h, bg, counting_attn))   
            h = torch.squeeze(torch.cat(all_h, dim=1))

        bg.ndata['h'] = h
        hg = dgl.mean_nodes(bg, 'h')
        return self.classify(hg)



class GWKClassifier_5hid(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, feat_drop_=0.,
                 attn_drop_=0.,):
        super(GWKClassifier_5hid, self).__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            nn.ModuleList([GWKLayer(in_dim, hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[0])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[0], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[1])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[1], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[2])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[2], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[3])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[3], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(1)])
        ])
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, bg, counting_attn, normalize = 'normal'):

        h = bg.in_degrees().view(-1, 1).float()
        num_nodes = h.shape[0]
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)

        for i, gnn in enumerate(self.layers):
            all_h = []
            for j, att_head in enumerate(gnn):
                all_h.append(att_head(h, bg, counting_attn))   
            h = torch.squeeze(torch.cat(all_h, dim=1))

        bg.ndata['h'] = h
        hg = dgl.mean_nodes(bg, 'h')
        return self.classify(hg)



class GWKClassifier_6hid(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, feat_drop_=0.,
                 attn_drop_=0.,):
        super(GWKClassifier_6hid, self).__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            nn.ModuleList([GWKLayer(in_dim, hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[0])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[0], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[1])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[1], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[2])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[2], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[3])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[3], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[4])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[4], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(1)])
        ])
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, bg, counting_attn, normalize = 'normal'):

        h = bg.in_degrees().view(-1, 1).float()
        num_nodes = h.shape[0]
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)

        for i, gnn in enumerate(self.layers):
            all_h = []
            for j, att_head in enumerate(gnn):
                all_h.append(att_head(h, bg, counting_attn))   
            h = torch.squeeze(torch.cat(all_h, dim=1))

        bg.ndata['h'] = h
        hg = dgl.mean_nodes(bg, 'h')
        return self.classify(hg)




class GWKClassifier_7hid(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, feat_drop_=0.,
                 attn_drop_=0.,):
        super(GWKClassifier_7hid, self).__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            nn.ModuleList([GWKLayer(in_dim, hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[0])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[0], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[1])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[1], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[2])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[2], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[3])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[3], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[4])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[4], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[5])]),
            
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[5], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(1)])
        ])
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, bg, counting_attn, normalize = 'normal'):

        h = bg.in_degrees().view(-1, 1).float()
        num_nodes = h.shape[0]
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)

        for i, gnn in enumerate(self.layers):
            all_h = []
            for j, att_head in enumerate(gnn):
                all_h.append(att_head(h, bg, counting_attn))   
            h = torch.squeeze(torch.cat(all_h, dim=1))

        bg.ndata['h'] = h
        hg = dgl.mean_nodes(bg, 'h')
        return self.classify(hg)




class GWKClassifier_8hid(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, feat_drop_=0.,
                 attn_drop_=0.,):
        super(GWKClassifier_8hid, self).__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            nn.ModuleList([GWKLayer(in_dim, hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[0])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[0], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[1])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[1], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[2])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[2], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[3])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[3], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[4])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[4], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[5])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[5], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[6])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[6], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(1)])
        ])
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, bg, counting_attn, normalize = 'normal'):

        h = bg.in_degrees().view(-1, 1).float()
        num_nodes = h.shape[0]
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)

        for i, gnn in enumerate(self.layers):
            all_h = []
            for j, att_head in enumerate(gnn):
                all_h.append(att_head(h, bg, counting_attn))   
            h = torch.squeeze(torch.cat(all_h, dim=1))

        bg.ndata['h'] = h
        hg = dgl.mean_nodes(bg, 'h')
        return self.classify(hg)





class GWKClassifier_9hid(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, feat_drop_=0.,
                 attn_drop_=0.,):
        super(GWKClassifier_9hid, self).__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            nn.ModuleList([GWKLayer(in_dim, hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[0])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[0], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[1])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[1], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[2])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[2], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[3])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[3], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[4])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[4], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[5])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[5], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[6])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[6], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[7])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[7], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(1)])
        ])
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, bg, counting_attn, normalize = 'normal'):

        h = bg.in_degrees().view(-1, 1).float()
        num_nodes = h.shape[0]
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)

        for i, gnn in enumerate(self.layers):
            all_h = []
            for j, att_head in enumerate(gnn):
                all_h.append(att_head(h, bg, counting_attn))   
            h = torch.squeeze(torch.cat(all_h, dim=1))

        bg.ndata['h'] = h
        hg = dgl.mean_nodes(bg, 'h')
        return self.classify(hg)






class GWKClassifier_10hid(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, feat_drop_=0.,
                 attn_drop_=0.,):
        super(GWKClassifier_10hid, self).__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            nn.ModuleList([GWKLayer(in_dim, hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[0])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[0], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[1])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[1], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[2])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[2], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[3])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[3], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[4])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[4], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[5])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[5], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[6])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[6], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[7])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[7], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(num_heads[8])]),
            nn.ModuleList([GWKLayer(hidden_dim * num_heads[8], hidden_dim, feat_drop = feat_drop_, attn_drop = attn_drop_, agg_activation=F.elu) for _ in range(1)])
        ])
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, bg, counting_attn, normalize = 'normal'):

        h = bg.in_degrees().view(-1, 1).float()
        num_nodes = h.shape[0]
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)

        for i, gnn in enumerate(self.layers):
            all_h = []
            for j, att_head in enumerate(gnn):
                all_h.append(att_head(h, bg, counting_attn))   
            h = torch.squeeze(torch.cat(all_h, dim=1))

        bg.ndata['h'] = h
        hg = dgl.mean_nodes(bg, 'h')
        return self.classify(hg)


##### Convolutional model definition

class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, normalize = 'normal'):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.in_degrees().view(-1, 1).float()
        
        #print('input shape is \n')
        #print(h.shape)
        num_nodes = h.shape[0]
        
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)
        #if std_:
        #   h = (h - mean_)/std_
        #else:   
        #   h = h/np.max(features)

        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)
    
    
    
# 1 hidden layer 
class GCNClassifier_1hid(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier_1hid, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim[0])
        self.classify = nn.Linear(hidden_dim[-1], n_classes)

    def forward(self, g, normalize = 'normal'):
        h = g.in_degrees().view(-1, 1).float()

        num_nodes = h.shape[0]
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)

        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))

        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)
        
        


# 2 hidden layers 
class GCNClassifier_2hid(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier_2hid, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim[0])
        self.conv2 = GraphConv(hidden_dim[0], hidden_dim[1])
        self.classify = nn.Linear(hidden_dim[-1], n_classes)

    def forward(self, g, normalize = 'normal'):
        h = g.in_degrees().view(-1, 1).float()

        num_nodes = h.shape[0]
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)

        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)
        
        
        
        
        
# 3 hidden layers 
class GCNClassifier_3hid(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier_3hid, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim[0])
        self.conv2 = GraphConv(hidden_dim[0], hidden_dim[1])
        self.conv3 = GraphConv(hidden_dim[1], hidden_dim[2])
        self.classify = nn.Linear(hidden_dim[-1], n_classes)

    def forward(self, g, normalize = 'normal'):
        h = g.in_degrees().view(-1, 1).float()

        num_nodes = h.shape[0]
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)

        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)
    
    

# 4 hidden layers 
class GCNClassifier_4hid(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier_4hid, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim[0])
        self.conv2 = GraphConv(hidden_dim[0], hidden_dim[1])
        self.conv3 = GraphConv(hidden_dim[1], hidden_dim[2])
        self.conv4 = GraphConv(hidden_dim[2], hidden_dim[3])
        
        self.classify = nn.Linear(hidden_dim[-1], n_classes)

    def forward(self, g, normalize = 'normal'):
        h = g.in_degrees().view(-1, 1).float()

        num_nodes = h.shape[0]
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)

        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.conv4(g, h))
        
        
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)
        
    
    

# 5 hidden layers 
class GCNClassifier_5hid(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier_5hid, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim[0])
        self.conv2 = GraphConv(hidden_dim[0], hidden_dim[1])
        self.conv3 = GraphConv(hidden_dim[1], hidden_dim[2])
        self.conv4 = GraphConv(hidden_dim[2], hidden_dim[3])
        self.conv5 = GraphConv(hidden_dim[3], hidden_dim[4])
        
        
        self.classify = nn.Linear(hidden_dim[4], n_classes)

    def forward(self, g, normalize = 'normal'):
        h = g.in_degrees().view(-1, 1).float()

        num_nodes = h.shape[0]
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)

        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.conv4(g, h))
        h = F.relu(self.conv5(g, h))
        
        
        
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)
        

# 6 hidden layers 
class GCNClassifier_6hid(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier_6hid, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim[0])
        self.conv2 = GraphConv(hidden_dim[0], hidden_dim[1])
        self.conv3 = GraphConv(hidden_dim[1], hidden_dim[2])
        self.conv4 = GraphConv(hidden_dim[2], hidden_dim[3])
        self.conv5 = GraphConv(hidden_dim[3], hidden_dim[4])
        self.conv6 = GraphConv(hidden_dim[4], hidden_dim[5])
        
        self.classify = nn.Linear(hidden_dim[5], n_classes)

    def forward(self, g, normalize = 'normal'):
        h = g.in_degrees().view(-1, 1).float()

        num_nodes = h.shape[0]
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)

        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.conv4(g, h))
        h = F.relu(self.conv5(g, h))
        h = F.relu(self.conv6(g, h))
        
        
        
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)
        

# 7 hidden layers 
class GCNClassifier_7hid(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier_7hid, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim[0])
        self.conv2 = GraphConv(hidden_dim[0], hidden_dim[1])
        self.conv3 = GraphConv(hidden_dim[1], hidden_dim[2])
        self.conv4 = GraphConv(hidden_dim[2], hidden_dim[3])
        self.conv5 = GraphConv(hidden_dim[3], hidden_dim[4])
        self.conv6 = GraphConv(hidden_dim[4], hidden_dim[5])
        self.conv7 = GraphConv(hidden_dim[5], hidden_dim[6])
        
        self.classify = nn.Linear(hidden_dim[-1], n_classes)

    def forward(self, g, normalize = 'normal'):
        h = g.in_degrees().view(-1, 1).float()

        num_nodes = h.shape[0]
        features = h.numpy().flatten()
        
        if normalize == 'normal':
            mean_ = np.mean(features)
            std_ = np.std(features)
            h = (h - mean_)/std_
        elif normalize == 'minmax':
            h = h/np.max(features)

        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.conv4(g, h))
        h = F.relu(self.conv5(g, h))
        h = F.relu(self.conv6(g, h))
        h = F.relu(self.conv7(g, h))
        
        
        
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)
        
