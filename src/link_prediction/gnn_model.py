import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import math

from utils_link_prediction import combine_embeddings
from sklearn.ensemble import RandomForestClassifier

class GCNLayer(nn.Module):
    def __init__(self, A, nin, nout, dropout, activation, batch_norm=None, bias=True):
        super(GCNLayer, self).__init__()
        self.A = A
        self.W = nn.Parameter(torch.rand(nin, nout, requires_grad=True))
        
        self.batch_norm = nn.BatchNorm1d(nout)
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
            
        if bias:
            self.bias = nn.Parameter(torch.Tensor(nout))
        else:
            self.bias = None
        self.activation = activation
        self.reset_parameters0()

    def reset_parameters0(self):
        nn.init.xavier_uniform_(self.W.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def reset_parameters1(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.bias is not None: 
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X):       
        h = X
        if self.dropout:
            h = self.dropout(h)

        h = torch.mm(torch.mm(self.A, h), self.W)

        if self.batch_norm:
            h = self.batch_norm(h)

        if self.bias is not None:
            h = h + self.bias

        if self.activation:
            h = self.activation(h)

        return h


class GATLayer(nn.Module):
    """
    Simple PyTorch Implementation of the Graph Attention layer.
    """

    def __init__(self, A, nin, nout, num_heads, dropout, concat, activation, batch_norm=None, bias=True, alpha=0.2):
        super(GATLayer, self).__init__()
        self.A = A
        self.in_features   = nin    # 
        self.out_features  = nout   # 
        self.num_heads = num_heads
        # self.device = devices if devices is not None else torch.device('cpu')
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_heads * nout))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(nout))
        else:
            self.bias = None
        self.activation = activation        # drop prob = 0.6
        self.batch_norm = nn.BatchNorm1d(nout)

        self.alpha         = alpha          # LeakyReLU with negative input slope, alpha = 0.2
        self.concat        = concat         # conacat = True for all layers except the output layer.

        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice 
        self.W = nn.Parameter(torch.zeros(size=(nin, nout * num_heads)))
        self.a = nn.Parameter(torch.zeros(size=(2*nout, num_heads)))
        nn.init.xavier_uniform_(self.a.data)
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters0()


    def reset_parameters0(self):
        nn.init.xavier_uniform_(self.W.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, X, edge_index=None):
        h = X
        # Linear Transformation
        h = torch.mm(h, self.W)
        N = h.size()[0]
        # Attention Mechanism (shared attention, therefore repeat h at two dimensions to make a adj-like mask)
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a))
        # (h * self.a).sum(dim=-1)

        adj = torch.zeros_like(self.A)
        row, col = edge_index
        adj[tuple((row,col))] = 1

        if self.num_heads == 1:
            e = e.squeeze(2)
        else:
            adj = (adj.unsqueeze(2)).repeat(1, 1, self.num_heads)
        
        # Masked Attention
        zero_vec  = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        alpha = attention[row,col]
        

        weight_mat = torch.zeros_like(e)
        alpha_mat = torch.where(adj > 0, attention, weight_mat)

        if self.dropout:
            attention = self.dropout(attention)
        h = torch.matmul(attention, h)

        if self.num_heads > 1:
            if self.concat:
                h = h.view(-1, self.num_heads * self.out_features)
            else:
                h = h.mean(dim=2)

        if self.batch_norm:
            h = self.batch_norm(h)

        if self.bias is not None:
            h = h + self.bias

        if self.activation:
            h = self.activation(h)

        return h, alpha

class Classifier(nn.Module):
    def __init__(self, classifier, nin, nout, L):
        super(Classifier, self).__init__()
        self.classifier = classifier
        if self.classifier == 'mlp':
            list_FC_layers = [ nn.Linear( nin//2**l , nin//2**(l+1) , bias=True ) for l in range(L-1) ]
            list_FC_layers.append(nn.Linear( nin//2**(L-1) , nout , bias=True ))
            self.classify = nn.ModuleList(list_FC_layers)
            self.L = L
            for m in self.classify.modules():
                self.weights_init(m)
        elif self.classifier == 'direct':
            self.classify = None
        else:
            self.classify = None
        self.n_class = nout
        self.activation1 = nn.Softmax(dim=1)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
    def forward(self, X, y=None, mode='training', clf=None):
        out = X
        if self.classifier == 'mlp':
            for i in range(len(self.classify)):
                out = self.classify[i](out)   
            out = self.activation1(out)
        elif self.classifier == 'direct':
            out = self.activation1(out)
        return out


class EdgePredict(nn.Module):
    def __init__(self, aggregator, classifier, nin, nclass, L):
        super(EdgePredict, self).__init__()
        self.aggregator = aggregator
        if aggregator != 'attent':
            if aggregator == 'concat':
                agg_nin = 2*nin
            else:
                agg_nin = nin
            self.classifier = Classifier(classifier, agg_nin, nclass, L)
        else:
            self.classifier = Classifier(None, 1, nclass, L)

    def forward(self, h, edges, y=None, mode='training', clf=None):
        if self.aggregator == 'hadamard':
            emb = h[edges[:, 0]] * h[edges[:, 1]]
        elif self.aggregator == 'avg':
            emb = (h[edges[:, 0]] + h[edges[:, 1]]) / 2
        elif self.aggregator == 'concat':
            emb_pos = torch.cat((h[edges[:, 0]], h[edges[:, 1]]), dim=1)
        out = self.classifier(emb, y=y, mode=mode, clf=clf)
        return out


class GCN(nn.Module):
    def __init__(self, A, nfeat, nhid, nclass, nlayers, 
                 activation, dropout, aggregator, classifier, L=1):
        super(GCN, self).__init__()
        
        if nlayers > 1:
            conv = [GCNLayer(A, nfeat, nhid, dropout, activation)]
            for i in range(nlayers-2):
                conv.append(GCNLayer(A, nhid, nhid, dropout, activation))
            conv.append(GCNLayer(A, nhid, nhid, dropout, None))
        else:
            conv = [GCNLayer(A, nfeat, nhid, dropout, None)]
        self.conv = nn.ModuleList(conv)
        
        self.edge_predictor = EdgePredict(aggregator, classifier, nhid, nclass, L)

    def forward(self, h, edge_index):
        for i in range(len(self.conv)):
            h  = self.conv[i](h)
        return h, None
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss


class HiCoEx(nn.Module):
    def __init__(self, A, nfeat, nhid, num_heads, nclass, nlayers, 
                 activation, dropout, aggregator, classifier, L=1):
        super(HiCoEx, self).__init__()
        if nlayers > 1:
            gatconv = [GATLayer(A, nfeat, nhid, num_heads, dropout, True, activation)]
            for i in range(nlayers-1):
                gatconv.append(GATLayer(A, nhid*num_heads, nhid*num_heads, num_heads, dropout, True, activation))
            gatconv.append(GATLayer(A, nhid, nhid, 1, dropout, True, None))
        else:
            gatconv = [GATLayer(A, nfeat, nhid, 1, dropout, True, None)]
        self.gatconv = nn.ModuleList(gatconv)

        self.edge_predictor = EdgePredict(aggregator, classifier, nhid, nclass, L)

    def forward(self, h, edge_index=None):
        for i in range(len(self.gatconv)):
            h, alpha  = self.gatconv[i](h, edge_index=edge_index)
        return h, alpha

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss


class HiCoEx_pyg(torch.nn.Module):
    def __init__(self, nfeat, nhid, num_heads, nclass, nlayers, 
                 activation, dropout, aggregator, classifier, L=1):
        super(HiCoEx_pyg, self).__init__()

        self.gatconv = GATConv(nfeat, nhid, heads=1, dropout=dropout, concat=True, negative_slope=0.5)
        if activation:
            self.activation = activation

        self.edge_predictor = EdgePredict(aggregator, classifier, nhid, nclass, L)

    def forward(self, h, edge_index):
        # Dropout before the GAT layer is used to avoid overfitting in small datasets like Cora.
        # One can skip them if the dataset is sufficiently large.
        h, alpha = self.gatconv(h, edge_index, return_attention_weights=True)
        return h, alpha

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss


class GCN_pyg(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, 
                 activation, dropout, aggregator, classifier, L=1):
        super(GCN_pyg, self).__init__()

        self.gcnconv = GCNConv(nfeat, nhid)
        if activation:
            self.activation = activation

        self.edge_predictor = EdgePredict(aggregator, classifier, nhid, nclass, L)

    def forward(self, h, edge_index):
        # Dropout before the GAT layer is used to avoid overfitting in small datasets like Cora.
        # One can skip them if the dataset is sufficiently large.
        h = self.gcnconv(h, edge_index)
        return h, None

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss


