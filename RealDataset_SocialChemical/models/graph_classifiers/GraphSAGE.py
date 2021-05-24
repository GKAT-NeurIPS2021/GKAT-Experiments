import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import SAGEConv, global_max_pool
from torch_geometric.utils import to_undirected, to_dense_adj, to_networkx

import torch
import torch.nn.functional as F
import torch_geometric
from deepwalk import OnlyWalk


class GraphSAGE(nn.Module):
    def __init__(self, dim_features, dim_target, config):
        super().__init__()
        #self.counter = 0

        num_layers = config['num_layers']
        dim_embedding = config['dim_embedding']
        self.aggregation = config['aggregation']  # can be mean or max

        if self.aggregation == 'max':
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_features if i == 0 else dim_embedding

            conv = SAGEConv(dim_input, dim_embedding)
            # Overwrite aggregation method (default is set to mean
            conv.aggr = self.aggregation

            self.layers.append(conv)

        # For graph classification
        self.fc1 = nn.Linear(num_layers * dim_embedding, dim_embedding)
        
        print("num layers:", num_layers)
        print("dim_embedding:", dim_embedding)
        
        self.fc2 = nn.Linear(dim_embedding, dim_target)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        '''
        torch.save(edge_index, "edge_index.pt")
        torch.save(x, "x.pt")
        torch.save(batch, 'batch.pt')
        #edge_index = torch.load('edge_index.pt')
        x = torch.load('x.pt')
        batch = torch.load('batch.pt')
        '''
        
        '''
        #t0 = time.time()
        adj_mat = to_dense_adj(edge_index)
        #adj_mat = torch_geometric.utils.to_scipy_sparse_matrix(edge_index)
        rw_trans = adj_mat[0] / (adj_mat[0].sum(0)[:,None])
        p_0 = adj_mat[0] / (adj_mat[0].sum(0)[:,None])
        for w in range(1, 4):
            rw_trans += torch.matmul( rw_trans, torch.transpose(p_0,0,1))
        coefs = rw_trans / torch.diag(rw_trans)[:,None]
        #print(time.time()-t0)
        '''
        
        '''
        t0 = time.time()
        dt = torch_geometric.data.Data()
        dt.x = x
        dt.edge_index = edge_index
        
        graph = to_networkx(dt)
        
        path_length = 4
        num_random_walk = 10
        p = 1
        q = 1
        stopping_prob = 0
        n2v = OnlyWalk.Node2vec_onlywalk(graph = graph, path_length=path_length, num_paths=num_random_walk, p=p, q=q, stop_prob = stopping_prob, with_freq_mat = True)
        counting_atten = torch.from_numpy(n2v.walker.freq_mat)
        print(time.time()-t0)
        '''
        #print(adj_mat.sum())
        
        #print("###")
        #print(data.batch.shape)
        #print(data.x.shape)
        #print(data.edge_index.shape)
        
        #print(x.shape)
        #print(x)
        #counter = 0
        
        #self.counter +=1
        
        #print(self.counter)

        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)


        x = torch.cat(x_all, dim=1)
        #print(data.batch)
        #print("###")
        #print("$$")
        #print(x.shape)
        x = global_max_pool(x, batch)
        #print(x)
        #print(x.shape)
        
        x = F.relu(self.fc1(x))
        #print(x.shape)
        
        x = self.fc2(x)
        #print(x.shape)
        
        return x
