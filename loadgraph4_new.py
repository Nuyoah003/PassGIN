import torch as th
import torch
import numpy as np
import dgl
from torch.utils.data import Dataset
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import sys
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

class GraphDataset(Dataset):
    def __init__(self, graph_list, label_list):
        self.graph_list = graph_list
        self.label_list = label_list
        self.list_len = len(graph_list)

    def __getitem__(self, index):
        return self.graph_list[index], self.label_list[index]

    def __len__(self):
        return self.list_len

def GetOneHotMap(element_list):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(np.array(element_list))
    print(integer_encoded)
    #  binary encoder
    onehot_encoded = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoded.fit_transform(integer_encoded)
    mapp = dict()
    for label, encoder in zip(element_list, onehot_encoded.tolist()):
        mapp[label] = np.array(encoder, dtype=np.float32)
    return mapp, integer_encoded.reshape(-1, ), onehot_encoded

def pass2graph(path):
    with open(path, 'rb') as fo:
        graph_data = pickle.load(fo)
    element_set = set([])
    for i in list(map(lambda x: set(x['node_label']), graph_data)):
        element_set = element_set.union(i)
    element_list = list(element_set)
    file=open("element_list_new_4_new_180w_parkmobile.txt",'w')
    file.write(str(element_list))
    node_label_map, node_label_id, node_one_hot = GetOneHotMap(element_list)
    file=open("node_label_map_parkmobile_new_4_new_180w.txt",'w')
    file.write(str(node_label_map))
    dgl_list = list([])
    label_list = list([])
    for idx in range(0,len(graph_data)):
        graph_dict=graph_data[idx]
        edge_start, edge_end = graph_dict['edge_start'],graph_dict['edge_end']
        node_label = list(map(lambda x: node_label_map[x], graph_dict['node_label']))
        g = dgl.DGLGraph()
        g.add_nodes(graph_dict['number_nodes'])
        node_label = np.array(node_label)
        g.ndata['feature'] = torch.tensor(node_label)
        edge_w = torch.tensor([*(graph_dict['edge_weight']), *(graph_dict['edge_weight'])])
        g.add_edges([*edge_start, *edge_end], [*edge_end, *edge_start])  ###双向图
        g.edata['w'] = edge_w
        dgl_list.append(g)
        # label_list.append(torch.tensor(dict3[graph_dict['graph_label']], dtype=torch.int32))
        label_list.append(int(graph_dict['graph_label']))
    train_set=GraphDataset(dgl_list,label_list)
    return train_set
def collate(samples):
    graph_list, label_list = map(list, zip(*samples))
    batched_graph = dgl.batch(graph_list)
    return batched_graph, torch.tensor(label_list)

if __name__=="__main__":
    graph = pass2graph('data/parkmobile_graph_new_4_new_180w.pkl')
