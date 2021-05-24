#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:08:52 2021

@author: hanlin
"""

num_heads = [8,1]
hidden_dim = [8]

feat_drop = 0.0
attn_drop = 0.0

agg_activation = F.elu

layers = nn.ModuleList([])

if len(hidden_dim)==0:
    layers.append( GWKLayer(dim_features, dim_target, feat_drop = feat_drop, attn_drop = attn_drop, agg_activation=agg_activation) )
else:
    
    for i in range(len(hidden_dim)):
        if i==0:
            num_heads_in = 1
            num_heads_out = num_heads[i]
            hid_dim_in = dim_features
            hid_dim_out = hidden_dim[i]
            
        else:
            num_heads_in = num_heads[i-1]   
            num_heads_out =num_heads[i]    
            hid_dim_in = hidden_dim[i-1]
            hid_dim_out = hidden_dim[i]
            
        h_list = nn.ModuleList()
        for _ in range(num_heads_out):
            h_list.append( GWKLayer(hid_dim_in * num_heads_in , hid_dim_out, feat_drop = feat_drop, attn_drop = attn_drop, agg_activation=agg_activation) )
        layers.append(h_list)
        
    
    num_heads_in = num_heads_out
    num_heads_out = 1
    hid_dim_in = hid_dim_out
    hid_dim_out = dim_target
    layers.append( nn.ModuleList([GWKLayer(hid_dim_in * num_heads_in , hid_dim_out * num_heads_out, feat_drop = feat_drop, attn_drop = attn_drop, agg_activation=agg_activation)] ) )    
        