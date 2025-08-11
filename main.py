import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import utils
import loadgraph4_new_18w_parkmobile
from model import GINClassifier
import numpy as np
import gc
import time
import tracemalloc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
MODEL_PATH = 'model_new_4_new_18w_parkmobile/'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def parse_arg():
    parser = argparse.ArgumentParser(description='GIN')
    parser.add_argument('--num-epochs', type=int, default=40,
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

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    train_set= loadgraph4_new_18w_parkmobile.pass2graph('data/parkmobile_graph_new_4_new_18w.pkl')
    train_data_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=loadgraph4_new_18w_parkmobile.collate)
    input_dim = train_set[0][0].ndata['feature'].size(1)
    output_dim = train_set[0][0].ndata['feature'].size(1)
    if torch.cuda.is_available():
        is_cuda = True
    model = GINClassifier(args.num_layers, args.num_mlp_layers, input_dim, args.hidden_dim, output_dim, args.feat_drop,
                          args.learn_eps, args.graph_pooling_type, args.neigh_pooling_type, args.final_drop, is_cuda).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    epoch_losses = []
    # start=time.time()
    for epoch in range(args.num_epochs):
        # tracemalloc.start()
        model.train()
        epoch_loss = 0
        for iter, (bg, label) in enumerate(train_data_loader):
            bg = bg.to(device)
            label=label.to(device)
            prediction = model(bg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_data_loader)
        print('Epoch{}, loss {:.4f}'.format(epoch, epoch_loss))
        if epoch%5==0:
            torch.save(model.state_dict(), MODEL_PATH + 'Model-graph-18-' + str(epoch) + '.kpl')


def main():
    # parameters
    args = parse_arg()
    # trian
    train(args)



if __name__ == '__main__':
    main()
