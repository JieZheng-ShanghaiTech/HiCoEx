import os
import time
import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils_link_prediction import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from gnn_model import GCN, HiCoEx, GCN_pyg, HiCoEx_pyg

import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
import ipdb

# from gpu_mem_track import MemTracker
# gpu_tracker = MemTracker()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def data_prepare(args, k=10, offset_18=None):
    # load graph data
    # hic = np.load('{}/{}/hic/{}.npy'.format(
    #     args.data_root, args.dataset, args.chromatin_network_name))
    adj = np.load('{}/{}/chromatin_networks/{}.npy'.format(
        args.data_root, args.dataset, args.chromatin_network_name))
    adj = np.triu(adj, 1)
    adj = adj + adj.T
    adj = np.nan_to_num(adj)
    # ipdb.set_trace()
    
    print('Use {} vector as node features'.format(args.node_feature))
    if args.node_feature == 'random':
        x_feat = torch.randn(adj.shape[0], int(adj.shape[1]))
    elif args.node_feature == 'one-hot':
        x_feat = torch.ones(adj.shape[0], adj.shape[1], requires_grad=True)
    else:
        x_feat = np.load(args.nfeat_path)
        x_feat = torch.FloatTensor(x_feat, requires_grad=True)

    return adj, x_feat
        

def view_model_param(args, model):
    total_param = 0
    print('{} MODEL DETAILS:\n'.format(args.method))
    print(model)
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', total_param)


def train_epoch(args, model, optimizer, train_batch, X_train, y_train, x_feat, edge_index):
    model.train()
    
    e_loss = 0.0
    num = 0
    scores = []
    y_label = []

    features = x_feat.to(args.device)

    if args.classifier == 'rf':
        h, _ = model.forward(features, edge_index)
        (scores, clf) = model.edge_predictor(h, X_train, y_train, mode='training')
        loss = model.loss(scores, y_train)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2, norm_type=2)
        optimizer.step()

        y_label = y_train.cpu().numpy()
        scores = scores.cpu().detach().numpy()

        y_out_prob = scores[:,1]
        y_out = scores.argmax(axis=1)

        results = {}
        results['auc'] = roc_auc_score(y_label, y_out_prob)
        results['acc'] = accuracy_score(y_label, y_out)
        results['f1'] = f1_score(y_label, y_out)
        results['precision'] = precision_score(y_label, y_out)
        results['recall'] = recall_score(y_label, y_out)
        results['predictions'] = y_out
        return (loss.item()/X_train.size(0), results, clf)
    else:
        for perm in train_batch:     
            batch_X = X_train[perm]
            batch_y = y_train[perm]

            h, _ = model.forward(features, edge_index)
            out = model.edge_predictor(h, batch_X)
            loss = model.loss(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2, norm_type=2)

            optimizer.step()

            e_loss += loss.detach().item()
            num += h.size(0)
            scores.append(out)
            y_label.append(batch_y)

        scores = torch.cat(scores, dim=0).cpu().detach().numpy()
        y_label = torch.cat(y_label, dim=0).cpu().numpy()
        y_out_prob = scores[:,1]
        y_out = scores.argmax(axis=1)

        # if np.isnan(y_out_prob).sum() != 0:
        #     print('{} {}'.format(args.dataset, args.chr_src))
        #     y_out_prob = np.zeros(y_label.shape)
    
        results = {}
        results['auc'] = roc_auc_score(y_label, y_out_prob)
        results['acc'] = accuracy_score(y_label, y_out)
        results['f1'] = f1_score(y_label, y_out)
        results['precision'] = precision_score(y_label, y_out)
        results['recall'] = recall_score(y_label, y_out)
        results['predictions'] = y_out
        return (e_loss/num, results)

def evaluate_network(args, model, X_test, y_test, x_feat, edge_index, clf=None):
    model.eval()

    with torch.no_grad():
        features = x_feat.to(args.device)
        h, _ = model.forward(features, edge_index)
        scores = model.edge_predictor(h, X_test, y_test, mode='testing', clf=clf)
        e_loss = model.loss(scores, y_test)
    
    scores = scores.cpu().numpy()
    y_test = y_test.cpu().numpy()
    y_out_prob = scores[:,1]
    y_out = scores.argmax(axis=1)
    
    # if np.isnan(y_out_prob).sum() != 0:
    #     print('{} {}'.format(args.dataset, args.chr_src))
    #     y_out_prob = np.zeros(y_test.shape)
    
    results = {}
    results['auc'] = roc_auc_score(y_test, y_out_prob)
    results['acc'] = accuracy_score(y_test, y_out)
    results['f1'] = f1_score(y_test, y_out)
    results['precision'] = precision_score(y_test, y_out)
    results['recall'] = recall_score(y_test, y_out)
    results['predictions'] = y_out
    return e_loss.item() / X_test.size(0), results, h


def train_main(args, X_train, y_train, X_test, y_test, filename, offset_18=None):
    log_dir = '{}/results/{}/chr_{}/{}_{}_{}'.format(args.data_root, args.dataset, args.chr_src, args.classifier, args.method, args.times)
    tb_dir = log_dir + '/tensorboard'
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=args.seed)
    X_train = torch.LongTensor(X_train).to(args.device)
    X_val = torch.LongTensor(X_val).to(args.device)
    X_test = torch.LongTensor(X_test).to(args.device)
    y_train = torch.LongTensor(y_train).to(args.device)
    y_val = torch.LongTensor(y_val).to(args.device)
    y_test = torch.LongTensor(y_test).to(args.device)
    n_classes = 2
    set_seed(args.seed)
    
    adj, x_feat = data_prepare(args, offset_18)
    
    edge_index = np.stack(np.where(adj==1.))
    edge_index = torch.LongTensor(edge_index).to(args.device)

    adj = torch.Tensor(adj).to(args.device)
    adj_hat = adj + torch.eye(adj.size(0), device=args.device)
    D = torch.diag(torch.sum(adj,1))
    D = torch.pow(D, -0.5)
    D[torch.isinf(D)] = 0
    adj_hat = torch.mm(torch.mm(D, adj_hat), D)

    print('Training on {} dataset with the graph of {} nodes:'.format(args.dataset, adj.shape[0]))
    print('{} training samples, {} validation samples, {} testing samples'.format(len(X_train), len(X_val), len(X_test)))

    if args.method == 'GNN_GCN':
        model = GCN(adj_hat, x_feat.size(1), 
            args.out_dim, n_classes, args.n_layers, 
            nn.ReLU(), args.dropout, args.aggregators, args.classifier)
    elif args.method == 'GNN_HiCoEx':
        model = HiCoEx(adj, x_feat.size(1), 
            args.out_dim, args.num_heads, n_classes, args.n_layers, 
            nn.ELU(), args.dropout, args.aggregators,args.classifier)
    elif args.method == 'GNN_GCN_pyg':
        from torch_sparse import SparseTensor
        edge_index = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(adj.shape[0], adj.shape[1]))
        model = GCN_pyg(x_feat.size(1), 
            args.out_dim, n_classes, args.n_layers, 
            nn.ReLU(), args.dropout, args.aggregators,args.classifier)
    elif args.method == 'GNN_HiCoEx_pyg':
        from torch_sparse import SparseTensor
        edge_index = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(adj.shape[0], adj.shape[1]))
        model = HiCoEx_pyg(x_feat.size(1), 
            args.out_dim, args.num_heads, n_classes, args.n_layers, 
            nn.ELU(alpha=0.8), args.dropout, args.aggregators,args.classifier)   

    view_model_param(args, model)
    
    train_batch = DataLoader(range(X_train.size(0)), args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=np.random.seed(args.seed))
    set_seed(args.seed)
    
    cpt_file = log_dir + '/best_model.pkl'
    if os.path.exists(cpt_file) and args.load_ckpt:
        model.load_state_dict(torch.load(cpt_file))

    model = model.to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=args.lr_reduce_factor,
                                                     patience=5,
                                                     verbose=True)

    best_loss = 999.
    best_epoch = 0.
    per_epoch_time = []
    
    with tqdm(range(args.epoches)) as t:
        if args.training:
            for epoch in t:

                t.set_description('Epoch {}'.format(epoch))

                start = time.time()
                result_tup = train_epoch(args, model, optimizer, train_batch, X_train, y_train, x_feat, edge_index)
                loss = result_tup[0]
                results = result_tup[1]
                if args.classifier == 'rf':
                    clf = result_tup[2]
                else:
                    clf = None
                    
                val_loss, val_results, h = evaluate_network(args, model, X_val, y_val, x_feat, edge_index, clf=clf)
                test_loss, test_results, h = evaluate_network(args, model, X_test, y_test, x_feat, edge_index, clf=clf)
                                
                if best_loss > val_loss:
                    best_loss = val_loss
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), cpt_file)
                # ipdb.set_trace()
                writer.add_scalar('train/_loss', loss, epoch)
                writer.add_scalar('train/_f1', results['f1'], epoch)
                writer.add_scalar('train/_precision', results['precision'], epoch)
                writer.add_scalar('train/_recall', results['recall'], epoch)
                writer.add_scalar('train/_acc', results['acc'], epoch)
                writer.add_scalar('val/_loss', val_loss, epoch)
                writer.add_scalar('val/_f1', val_results['f1'], epoch)
                writer.add_scalar('val/_precision', val_results['precision'], epoch)
                writer.add_scalar('val/_recall', val_results['recall'], epoch)
                writer.add_scalar('val/_acc', val_results['acc'], epoch)
                writer.add_scalar('test/_f1', test_results['f1'], epoch)
                writer.add_scalar('test/_precision', test_results['precision'], epoch)
                writer.add_scalar('test/_recall', test_results['recall'], epoch)
                writer.add_scalar('test/_acc', test_results['acc'], epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)   
                # ipdb.set_trace()
                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                                train_loss=loss, val_loss=val_loss,
                                train_P=results['precision'], train_R=results['recall'], train_acc=results['acc'], 
                                val_P=val_results['precision'], val_R=val_results['recall'], val_acc=val_results['acc'], 
                                test_P=test_results['precision'], test_R=test_results['recall'], test_acc=test_results['acc'])

                per_epoch_time.append(time.time()-start)
                if args.method == 'GNN_GAT_pyg':
                    scheduler.step(val_loss)
        
        model.load_state_dict(torch.load(cpt_file))
        
        clf = None
        final_loss, final_results, h = evaluate_network(args, model, X_test, y_test, x_feat, edge_index, clf=clf)
        print('The final test results: Precision {:.5f}   Recall {:.5f}   Accuracy{:5f}'.format(final_results['precision'], final_results['recall'], final_results['acc']))
        with open('{}/results/{}/{}'.format(args.data_root, args.dataset, filename), 'wb') as file_save:
            pickle.dump(final_results, file_save)
    return h