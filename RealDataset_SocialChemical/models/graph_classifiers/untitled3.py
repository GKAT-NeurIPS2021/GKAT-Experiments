#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:38:53 2021

@author: hanlin
"""


in_dim = 21
out_dim = 8
feat_drop = 0.0
attn_drop = 0.0
agg_activation=F.elu
alpha=0.2


feat_drop = nn.Dropout(feat_drop)
fc = nn.Linear(in_dim, out_dim, bias=False)
torch.nn.init.xavier_uniform_(fc.weight)
#torch.nn.init.zeros_(self.fc.bias)
attn_l = nn.Parameter(torch.ones(size=(out_dim, 1)))
attn_r = nn.Parameter(torch.ones(size=(out_dim, 1)))
attn_drop = nn.Dropout(attn_drop)
activation = nn.LeakyReLU(alpha)
softmax = nn.Softmax(dim = 1)

agg_activation=agg_activation



feat = x


h = feat_drop(feat)
head_ft = fc(h).reshape((h.shape[0], -1))

a1 = torch.mm(head_ft, attn_l)    # V x 1
a2 = torch.mm(head_ft, attn_r)     # V x 1
a = attn_drop(a1 + a2.transpose(0, 1))
a = activation(a)
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
     
            

class GWKLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_drop=0., attn_drop=0., alpha=0.2, agg_activation=F.elu):
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

    '''
    def clean_data(self):
        ndata_names = ['ft', 'a1', 'a2']
        edata_names = ['a_drop']
        for name in ndata_names:
            self.g.ndata.pop(name)
        for name in edata_names:
            self.g.edata.pop(name)
    '''
            
    def forward(self, feat, counting_attn):
      #with HiddenPrints():
        # prepare, inputs are of shape V x F, V the number of nodes, F the dim of input features
        #self.g = bg
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

