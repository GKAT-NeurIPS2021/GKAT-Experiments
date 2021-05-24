from torch_geometric import data


class Data(data.Data):
    def __init__(self,
                 x=None,
                 edge_index=None,
                 edge_attr=None,
                 y=None,
                 v_outs=None,
                 e_outs=None,
                 g_outs=None,
                 o_outs=None,
                 #laplacians=None,
                 #v_plus=None,
                 counting_attn = None,
                 #counting_attn_indices = None,
                 #counting_attn_values = None,
                 #counting_attn_size = None,
                 **kwargs):

        additional_fields = {
            'v_outs': v_outs,
            'e_outs': e_outs,
            'g_outs': g_outs,
            'o_outs': o_outs,
            #'laplacians': laplacians,
            #'v_plus': v_plus,
            'counting_attn': counting_attn,
            #'counting_attn_indices': counting_attn_indices,
            #'counting_attn_values': counting_attn_values,
            #'counting_attn_size': counting_attn_size
        }
        super().__init__(x, edge_index, edge_attr, y, **additional_fields)


class Batch(data.Batch):
    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        #laplacians = None
        v_plus = None
        
        #counting_attn_indices = None
        #counting_attn_values = None
        #counting_attn_size = None

        #print('len of datalist ########  ',(data_list[0]))
        
        
        #if 'laplacians' in data_list[0]:
        #    laplacians = [d.laplacians[:] for d in data_list]
        #    v_plus = [d.v_plus[:] for d in data_list]
        
        #print("good here 0")
        
        if 'counting_attn' in data_list[0]:
            counting_attn = [d.counting_attn['mat'] for d in data_list]
        
        #print("good here 1")
        
        if 'counting_attn_size' in data_list[0]:
            counting_attn_indices = [d.counting_attn_indices for d in data_list]
            counting_attn_values = [d.counting_attn_values for d in data_list]
            counting_attn_size = [d.counting_attn_size for d in data_list]

        #print("good here 2")

        copy_data = []
        for d in data_list:
            copy_data.append(Data(x=d.x,
                                  y=d.y,
                                  edge_index=d.edge_index,
                                  edge_attr=d.edge_attr,
                                  counting_attn = d.counting_attn,
                                  
                                  #counting_attn_indices = d.counting_attn_indices,
                                  #counting_attn_values = d.counting_attn_values,
                                  #counting_attn_size = d.counting_attn_size,

                                  v_outs=d.v_outs,
                                  g_outs=d.g_outs,
                                  e_outs=d.e_outs,
                                  o_outs=d.o_outs)
                             )
                             
        
        
        batch = data.Batch.from_data_list(copy_data, follow_batch=follow_batch)
        
                
                
        #batch['laplacians'] = laplacians
        #batch['v_plus'] = v_plus
        batch['counting_attn'] = counting_attn
        
        #batch['counting_attn_indices'] = counting_attn_indices
        #batch['counting_attn_values'] = counting_attn_values
        #batch['counting_attn_size'] = counting_attn_size


        
        return batch
