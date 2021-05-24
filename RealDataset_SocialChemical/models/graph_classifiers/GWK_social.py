import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import SAGEConv, global_max_pool, global_add_pool
from torch_geometric.utils import to_undirected, to_dense_adj, to_networkx, to_scipy_sparse_matrix

import torch_geometric
from deepwalk import OnlyWalk
import numpy as np





class GWKLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_drop=0., attn_drop=0., alpha=0.2, agg_activation=F.elu):
        super(GWKLayer, self).__init__()

        self.feat_drop = feat_drop  #nn.Dropout(feat_drop, training=self.training)
        self.attn_drop = attn_drop  #nn.Dropout(attn_drop)
        
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        #torch.nn.init.xavier_uniform_(self.fc.weight)
        #torch.nn.init.zeros_(self.fc.bias)
        self.attn_l = nn.Parameter(torch.ones(size=(out_dim, 1)))
        self.attn_r = nn.Parameter(torch.ones(size=(out_dim, 1)))
        self.activation = nn.LeakyReLU(alpha)
        self.softmax = nn.Softmax(dim = 1)

        self.agg_activation=agg_activation

            
    def forward(self, feat, counting_attn):
        h = F.dropout(feat, p=self.feat_drop, training=self.training)

        head_ft = self.fc(h).reshape((h.shape[0], -1))

        a1 = torch.mm(head_ft, self.attn_l)    # V x 1
        a2 = torch.mm(head_ft, self.attn_r)     # V x 1
        a = F.dropout(a1 + a2.transpose(0, 1), p=self.attn_drop, training=self.training)
        a = self.activation(a)

        maxes = torch.max(a, 1, keepdim=True)[0]
        a_ =  a - maxes
        #a = torch.clamp(a, min = -100, max = 100)
        
        a_nomi = torch.exp(a_) * counting_attn
        a_deno = torch.sum(a_nomi, 1, keepdim=True)
        a_nor = a_nomi/(a_deno+1e-9)

        ret = torch.mm(a_nor, head_ft)
        if self.agg_activation is not None:
            ret = self.agg_activation(ret)

        return ret







class GWKLayer_exp(nn.Module):
    def __init__(self, in_dim, out_dim, feat_drop=0., attn_drop=0., alpha=0.2, agg_activation=F.elu):
        super(GWKLayer_exp, self).__init__()

        self.feat_drop = feat_drop  #nn.Dropout(feat_drop, training=self.training)
        self.attn_drop = attn_drop  #nn.Dropout(attn_drop)
        
        self.fc_Q = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_K = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_V = nn.Linear(in_dim, out_dim, bias=False)
        
        #torch.nn.init.xavier_uniform_(self.fc_K.weight)
        #torch.nn.init.xavier_uniform_(self.fc_Q.weight)
        
        #torch.nn.init.zeros_(self.fc.bias)
        #self.attn_l = nn.Parameter(torch.ones(size=(out_dim, 1)))
        #self.attn_r = nn.Parameter(torch.ones(size=(out_dim, 1)))
        #self.activation = nn.LeakyReLU(alpha)
        self.softmax = nn.Softmax(dim = 1)

        self.agg_activation=agg_activation

            
    def forward(self, feat, counting_attn):
        h = F.dropout(feat, p=self.feat_drop, training=self.training)

        Q = self.fc_Q(h).reshape((h.shape[0], -1))
        K = self.fc_K(h).reshape((h.shape[0], -1))
        V = self.fc_V(h).reshape((h.shape[0], -1))
        
        logits = F.dropout( torch.matmul( Q, torch.transpose(K,0,1) ) , p=self.attn_drop, training=self.training) / np.sqrt(Q.shape[1])
        #logits = torch.clamp(logits, min = -100, max = 100)
        
        maxes = torch.max(logits, 1, keepdim=True)[0]
        logits =  logits - maxes
        
        #print(logits.max(), logits.min())
        
        a_nomi = torch.mul(torch.exp( logits  ), counting_attn)
        
        a_deno = torch.sum(a_nomi, 1, keepdim=True)
        a_nor = a_nomi/(a_deno+1e-9)

        ret = torch.mm(a_nor, V)
        if self.agg_activation is not None:
            ret = self.agg_activation(ret)

        return ret




class GWK_social(nn.Module):
    def __init__(self, dim_features, dim_target, config):
        super().__init__()    
        
        self.num_heads = config['num_heads']
        self.hidden_dim = config['hidden_dim']
        
        self.feat_drop = config['feat_drop']
        self.attn_drop = config['feat_drop'] #config['attn_drop']
        
        self.agg_activation = F.elu
        self.normalize = config['normalize']
        self.pooling = config['global_pooling']
        
        self.layers = nn.ModuleList([])
        
        self.dim_mid = config['dim_mid']
        
        self.fc_before_gat = config['fc_before_gat']
        #self.batch_forloop = config['batch_forloop']
        self.device = config['device']
        
        #self.fc2 = nn.Linear(dim_target, dim_target)
        
        self.fc1 = nn.Linear(self.dim_mid[0], self.dim_mid[1])
        self.fc2 = nn.Linear(self.dim_mid[1], dim_target)
        
        if self.fc_before_gat:
            self.fc_vertex = nn.Linear(dim_features, self.dim_mid[0])
        #torch.nn.init.xavier_uniform_(self.fc_vertex.weight)

        
        if len(self.hidden_dim)==0:
            if config['concat_exp'] == 'concat':
                self.layers.append( GWKLayer(self.dim_mid[0], self.dim_mid[0], feat_drop = self.feat_drop, attn_drop = self.attn_drop, agg_activation=self.agg_activation) )
            elif config['concat_exp'] == 'exp':
                self.layers.append( GWKLayer_exp(self.dim_mid[0], self.dim_mid[0], feat_drop = self.feat_drop, attn_drop = self.attn_drop, agg_activation=self.agg_activation) )
        else:
            
            for i in range(len(self.hidden_dim)):
                if i==0:
                    num_heads_in = 1
                    num_heads_out = self.num_heads[i]
                    hid_dim_out = self.hidden_dim[i]                        
                    
                    if self.fc_before_gat:
                        hid_dim_in = self.dim_mid[0]
                    else:
                        hid_dim_in = dim_features
                    
                else:
                    num_heads_in = self.num_heads[i-1]   
                    num_heads_out = self.num_heads[i]    
                    hid_dim_in = self.hidden_dim[i-1]
                    hid_dim_out = self.hidden_dim[i]
                    
                h_list = nn.ModuleList()
                for _ in range(num_heads_out):
                    if config['concat_exp'] == 'concat':
                        h_list.append( GWKLayer(hid_dim_in * num_heads_in , hid_dim_out , feat_drop = self.feat_drop, attn_drop = self.attn_drop, agg_activation=self.agg_activation) )
                    elif config['concat_exp'] == 'exp':
                        h_list.append( GWKLayer_exp(hid_dim_in * num_heads_in , hid_dim_out , feat_drop = self.feat_drop, attn_drop = self.attn_drop, agg_activation=self.agg_activation) )
                    
                self.layers.append(h_list)
                
            
            num_heads_in = num_heads_out
            num_heads_out = 1
            hid_dim_in = hid_dim_out
            hid_dim_out = self.dim_mid[0]
            self.layers.append( nn.ModuleList([GWKLayer(hid_dim_in * num_heads_in , hid_dim_out * num_heads_out, feat_drop = self.feat_drop, attn_drop = self.attn_drop, agg_activation=self.agg_activation)] ) )    
                



    def forward(self, data):
        
        if self.device == 'cpu':
            x, batch, counting_attn = data.x, data.batch, (data.counting_attn)
        elif self.device == 'cuda':
            x, batch, counting_attn = data.x.cuda(), data.batch.cuda(), data.counting_attn
        
        
        '''
        torch.save(x, "x.pt")
        torch.save(batch, 'batch.pt')
        torch.save(counting_attn, 'counting_attn.pt')
      
        x = torch.load('x.pt')
        counting_attn = torch.load('counting_attn.pt')
        batch = torch.load('batch.pt')
        '''

        #####
        if self.normalize == 'ones':
            pass
        elif self.normalize == 'normal':
            x = (x - x.mean(1)[:,None]) / ((x.std(1)[:,None])+1e-9)
        elif self.normalize == 'batchnorm':
            x = ((x - x.mean(0)) / (x.std(0)+1e-9))
        elif self.normalize == 'minmax':
            x = x / ((x.max(1).values - x.min(1).values)[:, None])
        elif self.normalize == 'batchminmax':
            x = 2*((x - x.min(0)[0]) / (x.max(0)[0] - x.min(0)[0]))-1
        
        if self.fc_before_gat:
            x = F.relu(self.fc_vertex(x))
            
            
            
        #####
        
        if self.device == 'cpu':
            counting_attn_diag = torch.zeros(x.shape[0], x.shape[0])
        elif self.device == 'cuda':
            counting_attn_diag = torch.zeros(x.shape[0], x.shape[0]).cuda()
            
        start_row = 0
        for bt in range(batch.max()):
            c_size = counting_attn[bt].shape[0]
            counting_attn_diag[start_row:start_row+c_size, start_row:start_row+c_size] = counting_attn[bt]
            start_row += c_size
    
        #x_bt = x

        for i, gnn in enumerate(self.layers):
            all_h = []
            for j, att_head in enumerate(gnn):
                all_h.append(att_head(x, counting_attn_diag))   
            x = torch.squeeze(torch.cat(all_h, dim=1))  

        if self.pooling == 'max':
            x_out = global_max_pool(x, batch)
        elif self.pooling == 'mean':
            x_out = global_add_pool(x, batch)


    

        x_out = F.relu(self.fc1(x_out))
        x_out = self.fc2(x_out)
        
        
        
        return x_out






