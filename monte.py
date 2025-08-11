import dgl
import torch as th
import torch
# from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dgl.data import MiniGCDataset
from dgl.nn.pytorch import GraphConv
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from sklearn.metrics import accuracy_score
# from read_graph import read_graphs_in_networkx,save_graphs_nx
import re
import time
import sys
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from random import sample
from sklearn.model_selection import train_test_split
import networkx as nx
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import argparse
from model import GINClassifier
import sys
from decimal import Decimal
import random
torch.set_printoptions(precision=300)
# decimal.getcontext().prec=200
# sys.setrecursionlimit(1000000)
# np.set_printoptions(threshold=sys.maxsize)

def parse_arg():
    parser = argparse.ArgumentParser(description='GIN')
    parser.add_argument('--num-epochs', type=int, default=3000,
                        help="number of training epochs (default: 1000)")
    parser.add_argument('--batch-size', type=int, default=128,
                        help="sizes of training batches (default: 64)")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate (default: 0.01)")
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for splitting the dataset into 10 (default: 0)")
    parser.add_argument('--num_layers', type=int, default=5,
                        help="number of layers INCLUDING the input one (default: 5)")
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help="number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.")
    parser.add_argument('--neigh-pooling-type', type=str, default="sum", choices=["sum", "mean", "max"],
                        help="Pooling for over neighboring nodes: sum, mean")
    parser.add_argument('--graph-pooling-type', type=str, default="sum", choices=["sum", "mean"],
                        help="Pooling for the graph: max, sum, mean")
    parser.add_argument('--num-tasks', type=int, default=1,
                        help="number of the  task for the framework")
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help="number of hidden units")
    parser.add_argument('--feat-drop', type=float, default=0.05,
                        help="dropout rate of the feature")
    parser.add_argument('--final-drop', type=float, default=0.05,
                        help="dropout rate of the prediction layer")
    parser.add_argument('--learn-eps', action="store_true",
                        help='Whether to learn the epsilon weighting for the center nodes.')
    args = parser.parse_args()
    return args


DEVICE = torch.device("cuda:0")
class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim,allow_zero_in_degree=True)  # 定义第一层图卷积
        self.conv2 = GraphConv(hidden_dim, hidden_dim,allow_zero_in_degree=True)  # 定义第二层图卷积
        self.conv3 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.classify = nn.Linear(hidden_dim, n_classes)  # 定义分类器


    def forward(self, g):
        """g表示批处理后的大图，N表示大图的所有节点数量，n表示图的数量
        """
        h = g.ndata["label"].view(-1, 95).float()  # [N, 1]
        # h=g.ndata["label"].float()
        # print("++++++++++")
        # print(h.shape)
        # 执行图卷积和激活函数
        h = F.relu(self.conv1(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv2(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h  # 将特征赋予到图的节点
        # 通过平均池化每个节点的表示得到图表示
        hg = dgl.mean_nodes(g, 'h')  # [n, hidden_dim]
        return self.classify(hg)  # [n, n_classes]

import os
import pickle

file_name='dict3_parkmobile_new_4_new_180w.txt'
file=open(file_name).read()
input_dim = len(eval(file))
input_dim = input_dim
output_dim = input_dim
if torch.cuda.is_available():
    is_cuda = True
    torch.cuda.manual_seed_all(0)
args = parse_arg()
model = GINClassifier(args.num_layers, args.num_mlp_layers, input_dim, args.hidden_dim, output_dim, args.feat_drop,
                          args.learn_eps, args.graph_pooling_type, args.neigh_pooling_type, args.final_drop, is_cuda)
state_dict=torch.load('model_new_4_new_180w_parkmobile/Model-graph-180-15.kpl')
model.load_state_dict(state_dict)
model.eval()
model.to(DEVICE)
def encode(password):
    dict2 = set()
    dict3 = {}
    for i in password:
        for j in i:
            dict2.add(j)
    num = 0
    for i in dict2:
        dict3[i] = num
        num += 1
    dict3['END'] = num
    return dict3

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
# password = preprocess('data/testword_rockyou.txt')
# dict3=encode(password)
file1=open('dict3_parkmobile_new_4_new_180w.txt','r')
dict3=eval(file1.read())
file=open('element_list_new_4_new_180w_parkmobile.txt','r')
element_list = eval(file.read())
file2=open("node_label_map_parkmobile_new_4_new_180w.txt",'r')
string1=file2.read().replace(' ','')
string1=string1.replace('\n','')
string1=string1.replace('array','np.array')
string1=string1.replace('float32','np.float32')
print(string1)
dict_onehot=eval(string1)
dict3_new={value:key for key,value in dict3.items()}
# dict_onehot, node_label_id, node_one_hot = GetOneHotMap(element_list)
# with open(os.path.join("output_gcn3", 'onehot.pickle'), 'rb') as f:
#     dict_onehot = pickle.load(f)
# with open(os.path.join("output_gcn3", 'word.pickle'), 'rb') as f:
#     dict3 = pickle.load(f)


new_onehot = {str(v): k for k, v in dict_onehot.items()}
# new_dict3 = {v : k for k, v in dict3.items()}

max_num_nodes=20
# # number_of_graphs = 10
# chunk_size_guesser = 8192
# lower_probability_threshold = 10**-9

class GraphDataset(Dataset):
    def __init__(self, graph_list):
        self.graph_list = graph_list
        # self.label_list = label_list
        self.list_len = len(graph_list)

    def __getitem__(self, index):
        return self.graph_list[index]

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

def init_graph():
    # list1=[]
    # list2=[]
    # list3=[]
    # for m in range(0, 1):
    #     list1.append(m)
    #     list2.append(m + 1)
    # u, v = th.tensor(list1), th.tensor(list2)
    # g = dgl.graph((u, v))
    # list3.append(dict_onehot["START"])
    # list3.append(dict_onehot["START"])
    # numpy_array = np.array(list3)
    # torch_list = torch.tensor(numpy_array)
    # g.ndata['label'] = torch_list
    # return g
    for j in range(1, 2):
        dict1={}
        list1 = []
        list2 = []
        list3 = []
        for m in range(0, j):
            list1.append(m)
            list2.append(m + 1)
        edge_start, edge_end = torch.tensor(list1), torch.tensor(list2)
        dict1['edge_start']=edge_start
        dict1['edge_end']=edge_end
        if j==1:
            dict1['edge_weight']=[1]
        if j == 1:
            list3.append("START")
            list3.append("START")
        dict1['node_label']=list3
        dict1['number_nodes']=j+1
        dict1['number_edge']=len(edge_start)
        # graph_list.append(dict1)
        # element_set = set([])
        # for i in list(map(lambda x: set(x['node_label']), graph_data)):
        #     element_set = element_set.union(i)
        # element_list = list(element_set)
        # file=open('element_list.txt','r')
        # element_list = eval(file.read())
        # node_label_map, node_label_id, node_one_hot = GetOneHotMap(element_list)
        dgl_list = list([])
        label_list = list([])
        graph_dict = dict1
        edge_start, edge_end = graph_dict['edge_start'], graph_dict['edge_end']
        node_label = list(map(lambda x: dict_onehot[x], graph_dict['node_label']))
        g = dgl.DGLGraph()
        g.add_nodes(graph_dict['number_nodes'])
        node_label=[list(item) for item in node_label]
        node_label = np.array(node_label)
        g.ndata['feature'] = torch.tensor(node_label)
        edge_w = torch.tensor([*(graph_dict['edge_weight']), *(graph_dict['edge_weight'])])
        g.add_edges([*edge_start, *edge_end], [*edge_end, *edge_start])  ###双向图
        g.edata['w'] = edge_w
        # dgl_list.append(g)
        # # label_list.append(int(graph_dict['graph_label']))
        # train_set = GraphDataset(dgl_list, label_list)
        # return train_set
        return g

import random
class GenDataset(torch.utils.data.Dataset):
    def __init__(self, password):
        self.data=password
        # random.shuffle(self.data)
        self.data_gen = self.get_data()

    def get_data(self):
        # for doc in self.data:
        #     batch_pw=doc
        batch_pw=self.data
        while len(batch_pw) > 0:
            yield batch_pw.pop()

    def __len__(self):
        return len(self.data * 10)

    def __getitem__(self, idx):
        return next(self.data_gen)
#
class MyDataset(Dataset):
    def __init__(self, X):
        self.X = X
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx]

def collate(samples):
    # 输入参数samples是一个列表
    # 列表里的每个元素是图和标签对，如[(graph1, label1), (graph2, label2), ...]
    # zip(*samples)是解压操作，解压为[(graph1, graph2, ...), (label1, label2, ...)]
    # graphs= map(list, zip(*samples))
    # dgl.batch 将一批图看作是具有许多互不连接的组件构成的大型图
    return dgl.batch(samples)
def batch_prob(prefix_graph):
    torch.cuda.empty_cache()
    dataset = GraphDataset(prefix_graph)
    data_loader = DataLoader(dataset, batch_size=len(prefix_graph),collate_fn=collate)
    # train_dataset = GenDataset(prefix_graph)
    # data_loader = DataLoader(train_dataset, batch_size=len(prefix_graph), shuffle=False,
    #                          collate_fn=collate)
    with torch.no_grad():
        for iter, (batchg) in enumerate(data_loader):
            # batchg=batchg.to(DEVICE)
            # pred = torch.softmax(model(batchg.to(DEVICE)), 1)
            temp_pred = model(batchg.to(DEVICE))
            pred=torch.softmax(temp_pred,1)
            # pred = torch.max(pred, 1)[1].view(-1)
            # pred = pred.detach().cpu().numpy().tolist()[0]
            return pred

def extract_pwd_from_node1(node_list):
    return map(lambda x: x[0], node_list)

def extract_pwd_from_node2(node_list):
    return map(lambda x: x[-1], node_list)

def get_keys(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return ''

def create_graph(string_list):
    list1=[]
    list2=[]
    list3=[]
    dict1={}
    special_string = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    i=''.join(string_list)
    j=len(string_list)+1
    for m in range(0, j):
        list1.append(m)
        list2.append(m + 1)
    edge_start, edge_end = torch.tensor(list1), torch.tensor(list2)
    dict1['edge_start'] = edge_start
    dict1['edge_end'] = edge_end
    if j == 1:
        dict1['edge_weight'] = [1]
    elif j == 2:
        dict1['edge_weight'] = [1, 1]
    else:
        list1 = []
        list1.append(1)
        list1.append(1)
        for index in range(2, len(edge_start)):
            if i[edge_start[index - 2]].isalpha() and i[edge_end[index - 2]].isalpha():
                list1.append()
            elif i[edge_start[index - 2]].isdigit() and i[edge_end[index - 2]].isdigit():
                list1.append()
            elif i[edge_start[index - 2]] in special_string and i[
                edge_end[index - 2]] in special_string:
                list1.append()
            elif (i[edge_start[index - 2]].isalpha() and i[edge_end[index - 2]].isdigit()) or (
                    i[edge_start[index - 2]].isdigit() and i[edge_end[index - 2]].isalpha()):
                list1.append()
            elif (i[edge_start[index - 2]].isalpha() and i[edge_end[index - 2]] in special_string) or (
                    i[edge_start[index - 2]] in special_string and i[edge_end[index - 2]].isalpha()):
                list1.append()
            elif (i[edge_start[index - 2]].isdigit() and i[edge_end[index - 2]] in special_string) or (
                    i[edge_start[index - 2]] in special_string and i[edge_end[index - 2]].isdigit()):
                list1.append()
                
        dict1['edge_weight'] = list1
    if j == 1:
        list3.append("START")
        list3.append("START")
    else:
        list3.append("START")
        list3.append("START")
        for m1 in range(0, j - 1):
            list3.append(i[m1])
    dict1['node_label'] = list3
    dict1['number_nodes'] = j + 1
    dict1['number_edge'] = len(edge_start)
    graph_dict = dict1
    edge_start, edge_end = graph_dict['edge_start'], graph_dict['edge_end']
    node_label = list(map(lambda x: dict_onehot[x], graph_dict['node_label']))
    g = dgl.DGLGraph()
    g.add_nodes(graph_dict['number_nodes'])
    node_label = np.array(node_label)
    g.ndata['feature'] = torch.tensor(node_label)
    edge_w = torch.tensor([*(graph_dict['edge_weight']), *(graph_dict['edge_weight'])])
    g.add_edges([*edge_start, *edge_end], [*edge_end, *edge_start])  ###双向图
    g.edata['w'] = edge_w
    return g

def next_nodes(astring, prob, prediction,c):
    torch.cuda.empty_cache()
    special_string = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    total_preds = np.array([Decimal(float(x)) for x in prediction]) * Decimal(prob) ##
    answer = []
    if c not in dict3.keys():
        return
    index_n=dict3[c]
    chain_prob=Decimal(total_preds[index_n])
    all_label = astring.ndata["feature"]
    temp_tensor=torch.from_numpy(dict_onehot[c])
    temp_tensor=temp_tensor.unsqueeze(0)
    all_label=torch.cat((all_label,temp_tensor),0)
    # all_label.append(torch.tensor(dict_onehot[char]))
    string_temp = ""
    for m in all_label:
        # ttt=m.detach().numpy()
        string_temp += new_onehot[str(m.detach().numpy())]
    string_temp = string_temp.strip('START')
    # for i in string_temp:
    i=string_temp
    j=len(i)+1
    dict1 = {}
    list1 = []
    list2 = []
    list3 = []
    for m in range(0, j):
        list1.append(m)
        list2.append(m + 1)
    edge_start, edge_end = torch.tensor(list1), torch.tensor(list2)
    dict1['edge_start'] = edge_start
    dict1['edge_end'] = edge_end
    if j == 1:
        dict1['edge_weight'] = [1]
    elif j == 2:
        dict1['edge_weight'] = [1, 1]
    else:
        list1 = []
        list1.append(1)
        list1.append(1)
        for index in range(2, len(edge_start)):
            if i[edge_start[index - 2]].isalpha() and i[edge_end[index - 2]].isalpha():
                list1.append()
            elif i[edge_start[index - 2]].isdigit() and i[edge_end[index - 2]].isdigit():
                list1.append()
            elif i[edge_start[index - 2]] in special_string and i[
                edge_end[index - 2]] in special_string:
                list1.append()
            elif (i[edge_start[index - 2]].isalpha() and i[edge_end[index - 2]].isdigit()) or (
                    i[edge_start[index - 2]].isdigit() and i[edge_end[index - 2]].isalpha()):
                list1.append()
            elif (i[edge_start[index - 2]].isalpha() and i[edge_end[index - 2]] in special_string) or (
                    i[edge_start[index - 2]] in special_string and i[edge_end[index - 2]].isalpha()):
                list1.append()
            elif (i[edge_start[index - 2]].isdigit() and i[edge_end[index - 2]] in special_string) or (
                    i[edge_start[index - 2]] in special_string and i[edge_end[index - 2]].isdigit()):
                list1.append()
        dict1['edge_weight'] = list1
    if j == 1:
        list3.append("START")
        list3.append("START")
    else:
        list3.append("START")
        list3.append("START")
        for m1 in range(0, j - 1):
            list3.append(i[m1])
    dict1['node_label'] = list3
    dict1['number_nodes'] = j + 1
    dict1['number_edge'] = len(edge_start)
    graph_dict = dict1
    edge_start, edge_end = graph_dict['edge_start'], graph_dict['edge_end']
    node_label = list(map(lambda x: dict_onehot[x], graph_dict['node_label']))
    g = dgl.DGLGraph()
    g.add_nodes(graph_dict['number_nodes'])
    node_label = np.array(node_label)
    g.ndata['feature'] = torch.tensor(node_label)
    edge_w = torch.tensor([*(graph_dict['edge_weight']), *(graph_dict['edge_weight'])])
    g.add_edges([*edge_start, *edge_end], [*edge_end, *edge_start])  ###双向图
    g.edata['w'] = edge_w
    # edge_w = torch.tensor([*(graph_dict['edge_weight']), *(graph_dict['edge_weight'])])
    # g.add_edges([*edge_start, *edge_end], [*edge_end, *edge_start])  ###双向图
    # g.edata['w'] = edge_w
    # new_node_id = temp.number_of_nodes() - 1
    # temp.ndata['label'][new_node_id] = torch.tensor(dict_onehot[char])
    # # temp.nodes[new_node_id].data['label'] = dict_onehot[char]
    # last_node_id = new_node_id - 1
    # temp.add_edges(last_node_id, new_node_id)
    # chain_pass = astring + char
    answer.append((g, chain_prob))
    return answer

def gen_node(node_list,c):
    if len(node_list) == 0:
        return
    pwd_list = list(extract_pwd_from_node1(node_list))
    predictions = batch_prob(pwd_list)
    # print(torch.sum(predictions, dim=1))
    node_batch =[]
    for i, cur_node in enumerate(node_list):
        astring,prob = cur_node
        temp=next_nodes(astring, prob, predictions[i].cpu().detach().numpy(),c)
        return temp[0][0],temp[0][1]

def guess(astring , prob ,c):
    temp1,temp2=gen_node([(astring, prob)],c)
    return temp1,temp2

# file_gen = open("gens_test_gcn3_12.txt", 'w')
guess_numbers=[[] for i in range(200000)]
num = 0
end_idx = 0
lower_probability_threshold = 10 ** -90
sample_size=1000

def probability_calculation():
    password=[]
    guess_nums_test=[]
    file_password=open('testword/testword_csdn.txt','r')
    # file_psm=open('psm_gin/sorted_rockyou_gin_origin_100_new_4_new_100w_128.txt','w')
    for i in file_password:
        flag=1
        data=i.strip()
        for j in data:
            if j not in dict3.keys():
                flag=0
                break
        if flag==1:
            password.append(data)
        # print(len(password))
        # if len(password)==10:
        #     break
    # password="password"
    print(len(password))
    dict_password={}

    # password=["justin"]
    num=0
    sum=0
    for j in password:
        # num+=1
        # if num==50:
        #     break
        # start = time.time()
        # time1=time.time()
        flag = 1
        for i in j:
            if flag==1:
                temp1,temp2=guess(astring=init_graph(),prob=1,c=i)
                flag=0
            else:
                temp1,temp2=guess(astring=temp1, prob=temp2, c=i)
        # time2=time.time()
        # print(time2-time1)

        guess_nums_test.append((j,temp2))
    return guess_nums_test

import bisect
def spinoff_node(node):
    prob = node[0]
    global num
    global guess_numbers
    global probs
    num += 1
    if prob == 0:
        return
    idx = bisect.bisect_left(probs, prob) - 1
    if idx >= 0:
        guess_numbers[idx].append(prob)

def next_node_fn(astring, prob, prediction):
    # all_label = astring.ndata["feature"]
    # string_temp = ""
    # for m in all_label:
    #     # ttt=m.detach().numpy()
    #     string_temp += new_onehot[str(m.detach().numpy())]
    # astring = string_temp.strip('START')
    if len(astring) > 0 and astring[-1] == "END":
        return [], True
    conditional_predictions = prediction
    total_preds = conditional_predictions * prob

    indexes = np.arange(len(total_preds))
    above_cutoff = total_preds > lower_probability_threshold
    above_indices = indexes[above_cutoff]
    probs_above = total_preds[above_cutoff]
    answer = []
    threshould = True
    for i, prob_above in enumerate(probs_above):
        char = dict3_new[above_indices[i]]
        char_pro = tuple((char, prob_above, conditional_predictions[i]))
        chain_pass = astring + char_pro
        answer.append(chain_pass)
    if len(answer) == 0:
        threshould = False
    return answer, threshould

def choose_next_node(node_list):
    total = sum(map(lambda x: x[-1], node_list))
    r = random.uniform(0, total)
    upto = 0
    for node in node_list:
        pwd = node[:-2]
        prob = node[-2]
        cond_prob = node[-1]
        if upto + cond_prob > r:
            return tuple([prob])+pwd
        upto += cond_prob

def sample(node_list):
    real_node_list1 = []
    special_node_list1 = []
    real_node_list2 = []
    special_node_list2 = []
    for node in node_list:
        if node == None:
            continue
        # all_label = node[1].ndata["feature"]
        # string_temp = ""
        # for m in all_label:
        #     string_temp += new_onehot[str(m.detach().numpy())]
        # pwd = string_temp.strip('START')
        pwd = node[1:]
        pwd_graph=create_graph(pwd)
        temp_node=(node[0],pwd_graph)
        if len(pwd) > 0 and len(pwd) <= 5 and pwd[-1] == "END":
            continue
        if len(pwd) == 0:
            real_node_list1.append(temp_node)
            real_node_list2.append(node)
        elif len(pwd) <= max_num_nodes and len(pwd) > 0:
            if pwd[-1] == "END":
                special_node_list1.append(temp_node)
                special_node_list2.append(node)
            else:
                real_node_list1.append(temp_node)
                real_node_list2.append(node)
        else:
            continue

    length = len(real_node_list1)
    if length == 0:
        return
    pwd_list = list(extract_pwd_from_node2(real_node_list1))
    predictions = batch_prob(pwd_list)
    real_node_list1.extend(special_node_list1)
    real_node_list2.extend(special_node_list2)
    next_nodes = []
    for i, cur_node in enumerate(real_node_list2):
        astring = cur_node[1:]
        prob = cur_node[0]
        pred = predictions[i].cpu().detach().numpy() if i < length else np.array([])
        poss_next, threshold = next_node_fn(astring, prob, pred)
        if len(poss_next) == 0 and threshold:
            spinoff_node(cur_node)
            continue
        next_nodes.append(choose_next_node(poss_next))
    if len(next_nodes) != 0:
        sample(next_nodes)

def seed(sample_size):
    sample_pw = []
    for i in range(sample_size):
        prob_t = tuple([1])
        sample_pw.append(prob_t)
    return sample_pw

guess_nums_test=probability_calculation()
pwds, probs=zip(*sorted(guess_nums_test, key=lambda x:x[1]))
for i in range(100):
    data=seed(sample_size)
    sample(data)

guessed_nums = list(map(lambda items: list(
            map(lambda x: 1/x, items)), guess_numbers))
out_guess_numbers = [0] * len(guess_numbers)
for i in range(len(guess_numbers)):
    out_guess_numbers[i] = sum(guessed_nums[i]) / num
for j in range(len(out_guess_numbers) - 1, 0, -1):
    out_guess_numbers[j - 1] += out_guess_numbers[j]

file1 = open("guess_cross/guess_num_parkmobile2csdn_gin_180w.txt", 'w')

for i in range(len(pwds), 0, -1):
    idx = i - 1
    file1.write(str(pwds[idx])+' '+str(probs[idx])+' '+str(out_guess_numbers[idx])+'\n')




