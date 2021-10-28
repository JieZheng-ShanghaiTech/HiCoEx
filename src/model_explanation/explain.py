import sys  
sys.path.append('..')
sys.path.append('../link_prediction')
import os
from link_prediction.train_GNN import data_prepare,view_model_param
from link_prediction.gnn_model import HiCoEx, HiCoEx_pyg
from link_prediction.utils import set_gpu
from link_prediction.utils_link_prediction import *
from AttExplainer import AttExplainer
from plotting import plot

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import random
import argparse

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    # load data
    args, filename = setup_filenames_and_folders(args, args.chr_src)
    coexpression, disconnected_nodes = load_coexpression(args, args.chromatin_network_name, '{}_{}'.format(args.chr_src, args.chr_tgt))

    set_seed(args.seed)
    edges, non_edges = get_edges(coexpression)
    X_train, X_test, y_train, y_test = build_dataset(args, edges, non_edges, coexpression.shape[0])
    
    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))
    coexpression = np.triu(coexpression, 1)
    coexpression = coexpression + coexpression.T
    coexpression = np.nan_to_num(coexpression)

    n_classes = 2
    set_seed(args.seed)

    adj, x_feat = data_prepare(args)
    edge_index = np.stack(np.where(adj > 0))
    
    # statistic propotion of (adj, coep)
    row, col = X.T
    ctct = adj[row,col]
    id11 = np.where((ctct==1)&(y==1))[0]
    info = pd.read_csv('{}/{}/rna/expression_info_chr_{}.csv'.format(args.data_root, args.dataset, args.chr_src))
    name = info['Gene name'].tolist()   
    dist_tss = abs(info.loc[edge_index[0,:],'Transcription start site (TSS)'].values-info.loc[edge_index[1,:],'Transcription start site (TSS)'].values)
         
    X_exp = X[id11,:]
    y_exp = y[id11]

    x_feat = x_feat.to(args.device)
    adj = torch.Tensor(adj).to(args.device) 
    edge_index = torch.LongTensor(edge_index).to(args.device)
    
    if args.method == 'GNN_HiCoEx':
        model = HiCoEx(adj, x_feat.size(0), 
            args.out_dim, args.num_heads, n_classes, args.n_layers, 
            nn.ELU(), args.dropout, args.aggregators,args.classifier)
    elif args.method == 'GNN_HiCoEx_pyg':
        model = HiCoEx_pyg(x_feat.size(0), 
            args.out_dim, args.num_heads, n_classes, args.n_layers, 
            nn.ELU(), args.dropout, args.aggregators,args.classifier)

    model.embedding_size = args.out_dim
    
    view_model_param(args, model)
    log_dir = '{}/results/{}/test/chr_{}/{}_{}_{}'.format(args.data_root, args.dataset, args.chr_src, args.classifier, args.method, args.times)
    ckpt_dir = log_dir + '/best_model.pkl'
    print('loading model from '+ckpt_dir)
    checkpoint = torch.load(ckpt_dir)
    model.load_state_dict(checkpoint)
    model = model.to(args.device)

    
    expl_log_dir = log_dir + '/Attexplainer'
    if not os.path.exists(expl_log_dir) :
        os.makedirs(expl_log_dir)
    if not os.path.exists(expl_log_dir+'/statistics'):
        os.makedirs(expl_log_dir+'/statistics')
    if not os.path.exists(expl_log_dir+'/subgraphs'):
        os.makedirs(expl_log_dir+'/subgraphs')
    

    # th0,th1 = 99,98 #chr19 AD th0 95 th1 90, Isdia th0 99 th1 98
    num = int(len(X_test)/2)
    # num = len(X_test)
    
    explainer = AttExplainer(model, edge_index, x_feat)
    if args.glob:
        glob_alpha, graph, glob_graph, org_pred, glob_pred = explainer.explain(X_test[:num], args.th_glob, 'glob')
        org_out = org_pred.argmax(1).cpu().numpy()
        org_acc = accuracy_score(y_test[:num], org_out)

        glob_out = glob_pred.argmax(1).cpu().numpy()
        glob_acc = accuracy_score(y_test[:num], glob_out)

        final_results = {}
        final_results['org_acc'] = org_acc
        # final_results['att_sparsity'] = 1-glob_graph.size(1)/edge_index.size(1)
        final_results['att_sparsity'] = 1-torch.unique(glob_graph).size(0)/adj.size(0)
        final_results['att_infidelity_acc'] = org_acc-glob_acc
        final_results['att_infidelity_prob'] = (org_pred[:,1]-glob_pred[:,1]).mean()
        
        print('original prediction accuracy: %f' %final_results['org_acc'])
        print('Attention: Sparsity: %f' %final_results['att_sparsity'])
        print('Attention: Infidelity accuracy: %f' %final_results['att_infidelity_acc'])
        print('Attention: Infidelity prob: %f' %final_results['att_infidelity_prob'])

        labels = coexpression[edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()]
        # normalize the attention scores for plotting
        glob_alpha = glob_alpha.detach().cpu().numpy()
        norm_gaph = (glob_alpha/glob_alpha.min())/(glob_alpha.max()/glob_alpha.min())
        glob_th = np.percentile(norm_gaph, args.th_glob)
        print('plot global subgraph for all attention... ')
        plt_num = 120
        _ = plot(edge_index, norm_gaph, labels, 
                glob_th, plt_num, args.dataset, expl_log_dir, 
                name=name, args=args, show=False)

        exalnn = glob_graph.reshape(-1,).tolist()
        exaln = np.unique(exalnn)
        exald = info.loc[exaln]
        exald.to_csv('{}/statistics/global_att_graphnode.csv'.format(expl_log_dir))

        al05_id=np.where(norm_gaph>=0.2)[0]
        # plt.scatter(dist_tss[al05_id],norm_al[al05_id])
        # plt.show()
    
    if args.local:
        all_set = [[] for i in range(22)]
        all_set[args.chr_src-1].extend(args.gene_list)

        name_set = all_set[int(args.chr_src)-1]
        bio_set = [name.index(i) for i in name_set if i in name]
        if len(bio_set) is 0:
            print('The genes to be explained are not in our dataset, please check in the gene expression file or change genes')
        else:
            for g in bio_set:
                edge_list, y_list = [], []
                idx = np.stack(np.where(X_exp == g))[0]
                edge_list.append(X_exp[idx])
                y_list.append(y_exp[idx])
                edge_list = np.vstack(edge_list)
                y_list = np.hstack(y_list)
                
                att_node_list = []
                for i in range(0, len(edge_list)):
                    
                    pair = edge_list[i,:]
                    pair_y = y_list[i]
                    local_alpha, union_graph, local_graph, _, local_pred = explainer.explain(pair, args.th_local, 'local')
                    sg_idx = torch.where(union_graph[0]<union_graph[1])[0] # since the input graph is bidirectional (PyG accepted form) 
                    union_graph = union_graph[:,sg_idx]
                    local_alpha = local_alpha[sg_idx]
                    
                    union_graph = union_graph.cpu().numpy()
                    row, col = union_graph[0], union_graph[1]
                    labels = coexpression[row, col]
                    # normalize the attention scores on this subgraph for plotting
                    local_alpha = local_alpha.detach().cpu().numpy()
                    norm_laph = (local_alpha/local_alpha.min())/(local_alpha.max()/local_alpha.min())
                    local_th = np.percentile(norm_laph, args.th_local)
                    print('plot subgraph for pair: {}, {} ... '.format(pair[0], pair[1]))
                    plt_num = 100 #180 for islet heal; 70 for islet mining
                    node_11 = plot(union_graph, norm_laph, labels, 
                                    local_th, plt_num, args.dataset, expl_log_dir, idx=pair, 
                                    idx_adj=adj[pair[0], pair[1]].item(), idx_label=pair_y, 
                                    name=name, args=args, show=False)
                    
                    if node_11 is not None:
                        ald = info.loc[node_11]
                        ald.to_csv('{}/statistics/local_att_graphnode_{}_{}.csv'.format(expl_log_dir,pair[0],pair[1]))
    # for plotting the global attention scores vs. edge genomic distance
    if args.glob:
        return dist_tss, norm_gaph

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='../../data')
    parser.add_argument('--node-feature', type=str, default='random', 
                        choices=['random', 'one-hot', 'biological', 'pre-trained'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nfeat-path', type=str, default=None, 
                        help='require when node feature is biological or pre-trained')
    parser.add_argument('--dataset', type=str, default='adrenal_gland')
    parser.add_argument('--chr-src', type=int, default=1)
    parser.add_argument('--chr-tgt', type=int, default=1)
    parser.add_argument('--n-iter', type=int, default=1)
    parser.add_argument('--method', type=str, default='node2vec',
                        choices=['random', 'distance', 'topological', 'svd', 'node2vec', 
                                'GNN_GCN', 'GNN_HiCoEx', 'GNN_GCN_pyg', 'GNN_HiCoEx_pyg'])
    parser.add_argument('--chromatin-network-name', type=str)

    # params of trained model
    parser.add_argument('--coexp-thr', type=str, default=90.0)
    parser.add_argument('--emb-size', type=int, default=16)
    parser.add_argument('--aggregators', nargs='*', default=['hadamard'], choices=['hadamard', 'avg', 'l1', 'l2', 'attent'])
    parser.add_argument('--classifier', default='rf', choices=['mlp', 'direct', 'lr', 'svm', 'rf', 'random'])
    parser.add_argument('--n-layers', type=int, default=1)
    parser.add_argument('--out-dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num-heads', type=int, default=1)

    # params of explanier
    parser.add_argument('--test', default=True, action='store_true')
    parser.add_argument('--gene-list', type=str, nargs='+')
    parser.add_argument('--times', type=int, default=0)
    parser.add_argument('--gpu', default=False)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--glob', default=False)
    parser.add_argument('--local', default=False)
    parser.add_argument('--th-glob', type=int, default=70)
    parser.add_argument('--th-local', type=int, default=70)

    args = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args, _ = get_params()
    device = set_gpu(args.gpu, args.gpu_id)
    args.device = device
    main(args)