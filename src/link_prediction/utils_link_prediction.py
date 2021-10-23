import itertools
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool
from time import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sps
from bionev.utils import load_embedding
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils import intra_mask


def select_classifier(classifier_name, clf_params, seed=42):
    if classifier_name == 'mlp':
        classifier = MLPClassifier(max_iter=500, hidden_layer_sizes=(100,))
    elif classifier_name == 'svm':
        classifier = SVC(gamma='scale')
    elif classifier_name == 'rf':
        classifier = RandomForestClassifier(n_jobs=10, **clf_params)
    elif classifier_name == 'random':
        classifier = classifier_name
    else:
        classifier = LogisticRegression(solver='lbfgs')
    return classifier


def confusion_matrix_distinct(y_true, y_pred, ids, mask):
    is_intra = mask[ids[:, 0], ids[:, 1]].astype(bool)
    y_true_intra, y_pred_intra = y_true[is_intra], y_pred[is_intra]
    print('Intra accuracy: ', accuracy_score(y_true_intra, y_pred_intra))
    print(confusion_matrix(y_true_intra, y_pred_intra))

    if (is_intra == 0).any():
        y_true_inter, y_pred_inter = y_true[~is_intra], y_pred[~is_intra]
        print('Inter accuracy: ', accuracy_score(y_true_inter, y_pred_inter))
        print(confusion_matrix(y_true_inter, y_pred_inter))


def evaluate(X_train, y_train, X_test, y_test, classifier, mask):
    ids = X_test[:, :2].astype(int)
    if classifier == 'random':
        y_pred = np.random.randint(0, 2, size=y_test.shape)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[:, 2:], y_train)
        X_test_scaled = scaler.transform(X_test[:, 2:])

        start = time()
        classifier.fit(X_train_scaled, y_train)
        y_pred = classifier.predict(X_test_scaled)
        end = time()

    if mask is not None:
        confusion_matrix_distinct(y_test, y_pred, ids, mask)

    results = {}
    results['roc'] = roc_auc_score(y_test, y_pred)
    results['acc'] = accuracy_score(y_test, y_pred)
    results['f1'] = f1_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['recall'] = recall_score(y_test, y_pred)
    results['predictions'] = list(y_pred)
    return results


def evaluate_embedding(X_train, y_train, classifier_name, verbose=True, clf_params={}, cv_splits=5, mask=None, X_test=None,
                       y_test=None):
    results = defaultdict(list)
    if X_test is None and y_test is None:
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True)
        for train_index, val_index in skf.split(X_train, y_train):
            classifier = select_classifier(classifier_name, clf_params)
            results_iter = evaluate(X_train[train_index], y_train[train_index], X_train[val_index],
                                    y_train[val_index], classifier, mask)
            if verbose:
                print("Accuracy:", results_iter['acc'], "- ROC:", results_iter['roc'], "- F1:", results_iter['f1'],
                      "- Precision:", results_iter['precision'], "- Recall", results_iter['recall'])

            for key in results_iter.keys():
                results[key].append(results_iter[key])
    else:
        classifier = select_classifier(classifier_name, clf_params)
        results = evaluate(X_train, y_train, X_test, y_test, classifier, mask)
        if verbose:
            print("Accuracy:", results['acc'], "- ROC:", results['roc'], "- F1:", results['f1'],
                  "- Precision:", results['precision'], "- Recall", results['recall'])
    return results


def generate_embedding(args, emb_path, interactions_path, command):
    os.makedirs('{}/{}/embeddings/{}/'.format(args.data_root, args.dataset, args.method.lower()), exist_ok=True)
    if not os.path.exists(
            '{}/{}/embeddings/{}/{}.npy'.format(args.data_root, args.dataset, args.method.lower(), emb_path)) or args.force:
        adj = np.load(interactions_path)
        graph = from_numpy_matrix(adj)

        nx.write_weighted_edgelist(graph, '{}/{}/chromatin_networks/{}.edgelist'.format(args.data_root, args.dataset, args.name))

        print(command)
        os.system(command)
        emb_dict = load_embedding(
            '{}/{}/embeddings/{}/{}.txt'.format(
                args.data_root, args.dataset, args.method.lower(), emb_path))

        emb = np.zeros((adj.shape[0], args.emb_size))

        disconnected_nodes = []

        print('N. genes', adj.shape[0])
        for gene in range(adj.shape[0]):
            try:
                emb[gene, :] = emb_dict[str(gene)]
            except KeyError:
                print('Node', gene, 'disconnected.')
                # np.delete(emb, i, axis=0)
                emb[gene, :] = np.nan
                disconnected_nodes.append(gene)

        os.makedirs('{}/{}/disconnected_nodes/'.format(args.data_root, args.dataset), exist_ok=True)
        np.save(
            '{}/{}/disconnected_nodes/{}.npy'.format(
                args.data_root, args.dataset, args.name), np.array(disconnected_nodes))

        if args.save_emb:
            np.save('{}/{}/embeddings/{}/{}.npy'.format(args.data_root, args.dataset, args.method.lower(), emb_path), emb)
        os.remove('{}/{}/embeddings/{}/{}.txt'.format(args.data_root, args.dataset, args.method.lower(), emb_path))
        os.remove('{}/{}/chromatin_networks/{}.edgelist'.format(args.data_root, args.dataset, args.name))
        return emb


def from_numpy_matrix(A):
    # IMPORTANT: do not use for the co-expression matrix, otherwise the nans will be ignored and considered as non_edges
    A[np.isnan(A)] = 0

    if A.shape[0] != A.shape[1]:
        graph = nx.algorithms.bipartite.from_biadjacency_matrix(sps.csr_matrix(A))
    else:
        graph = nx.from_numpy_array(A)
    return graph


def distance_embedding(path, dataset, edges, non_edges, chr_src=None):
    if chr_src is None:
        gene_info = pd.read_csv(
            '{}/{}/rna/expression_info_chr_all.csv'.format(
                path, dataset))
    else:
        gene_info = pd.read_csv(
            '{}/{}/rna/expression_info_chr_{}_rna.csv'.format(
                path, dataset, chr_src))

    pos_distances = np.abs(gene_info.iloc[edges[:, 0]]['Transcription start site (TSS)'].to_numpy() -
                           gene_info.iloc[edges[:, 1]]['Transcription start site (TSS)'].to_numpy())

    neg_distances = np.abs(gene_info.iloc[non_edges[:, 0]]['Transcription start site (TSS)'].to_numpy() -
                           gene_info.iloc[non_edges[:, 1]]['Transcription start site (TSS)'].to_numpy())

    pos_features = np.hstack((edges, pos_distances[:, None]))
    neg_features = np.hstack((non_edges, neg_distances[:, None]))
    X = np.vstack((pos_features, neg_features))
    return X


def add_topological_edge_embeddings(graph_hic, edges, non_edges, features_pos, features_neg):
    shortest_path_lengths_pos = np.array(list(
        map(lambda e: nx.shortest_path_length(graph_hic, e[0], e[1]) if nx.has_path(graph_hic, e[0],
                                                                                    e[1]) else np.nan,
            edges)))
    shortest_path_lengths_neg = np.array(list(
        map(lambda e: nx.shortest_path_length(graph_hic, e[0], e[1]) if nx.has_path(graph_hic, e[0],
                                                                                    e[1]) else np.nan,
            non_edges)))

    jaccard_index_pos = np.array(list(map(lambda e: e[2], nx.jaccard_coefficient(graph_hic, edges))))
    jaccard_index_neg = np.array(list(map(lambda e: e[2], nx.jaccard_coefficient(graph_hic, non_edges))))

    features_pos = np.hstack((features_pos, shortest_path_lengths_pos[:, None], jaccard_index_pos[:, None]))
    features_neg = np.hstack((features_neg, shortest_path_lengths_neg[:, None], jaccard_index_neg[:, None]))
    return features_pos, features_neg


def topological_features(args, edges, non_edges):
    adj_hic = np.load('{}/{}/chromatin_networks/{}.npy'.format(args.data_root, args.dataset, args.chromatin_network_name))
    graph_hic = from_numpy_matrix(adj_hic)
    graph_hic = nx.convert_node_labels_to_integers(graph_hic)

    if os.path.exists('{}/{}/embeddings/topological/{}.npy'.format(args.data_root, args.dataset, args.chromatin_network_name)):
        embeddings = np.load('{}/{}/embeddings/topological/{}.npy'.format(args.data_root, args.dataset, args.chromatin_network_name))
    else:
        degrees = np.array(list(dict(graph_hic.degree()).values()))
        betweenness = np.array(list(betweenness_centrality_parallel(graph_hic, 20).values()))
        clustering = np.array(list(nx.clustering(graph_hic).values()))

        embeddings = np.hstack((degrees[:, None], betweenness[:, None], clustering[:, None]))

        os.makedirs('{}/{}/embeddings/topological/'.format(args.data_root, args.dataset), exist_ok=True)
        np.save('{}/{}/embeddings/topological/{}.npy'.format(args.data_root, args.dataset, args.chromatin_network_name), embeddings)

    features_pos, features_neg = combine_embeddings(embeddings, args.aggregators, edges, non_edges)
    features_pos, features_neg = add_topological_edge_embeddings(graph_hic, edges, non_edges, features_pos,
                                                                 features_neg)
    X = np.vstack((features_pos, features_neg))

    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=9999)
    X = imp.fit_transform(X)
    return X


def combine_embeddings(embeddings, aggregators, edges, non_edges):
    # Add edges and non_edges ids in dataset to identify the type of interaction for the confusion matrix
    # They will be removed from the dataset before training
    pos_features = edges
    neg_features = non_edges
    if 'hadamard' in aggregators:
        pos_features, neg_features = hadamard_embedding(pos_features, neg_features, embeddings, edges, non_edges)
    if 'avg' in aggregators:
        pos_features, neg_features = average_embedding(pos_features, neg_features, embeddings, edges, non_edges)
    if 'l1' in aggregators:
        pos_features, neg_features = l1_embedding(pos_features, neg_features, embeddings, edges, non_edges)
    
    if pos_features.shape[1] == 2 or neg_features.shape[1] == 2:
        raise ValueError('No aggregator defined.')

    return pos_features, neg_features


def method_embedding(args, n_nodes, edges, non_edges):
    if args.method == 'random':
        embeddings = np.random.rand(n_nodes, args.emb_size)
    else:
        embeddings = np.load(
            '{}/{}/embeddings/{}/{}.npy'.format(args.data_root, args.dataset, args.method, args.embedding))

    features_pos, features_neg = combine_embeddings(embeddings, args.aggregators, edges, non_edges)
    X = np.vstack((features_pos, features_neg))
    return X


def append_features(pos_features, neg_features, pos_features_partial, neg_features_partial):
    if pos_features is None or neg_features is None:
        pos_features = pos_features_partial
        neg_features = neg_features_partial
    else:
        pos_features = np.hstack((pos_features, pos_features_partial))
        neg_features = np.hstack((neg_features, neg_features_partial))
    return pos_features, neg_features


def hadamard_embedding(pos_features, neg_features, embeddings, edges, non_edges):
    pos_features_partial = embeddings[edges[:, 0]] * embeddings[edges[:, 1]]
    neg_features_partial = embeddings[non_edges[:, 0]] * embeddings[non_edges[:, 1]]
    return append_features(pos_features, neg_features, pos_features_partial, neg_features_partial)


def average_embedding(pos_features, neg_features, embeddings, edges, non_edges):
    pos_features_partial = np.array(
        list(map(lambda edge: np.mean((embeddings[edge[0]], embeddings[edge[1]]), axis=0), edges)))
    neg_features_partial = np.array(
        list(map(lambda edge: np.mean((embeddings[edge[0]], embeddings[edge[1]]), axis=0), non_edges)))
    return append_features(pos_features, neg_features, pos_features_partial, neg_features_partial)


def l1_embedding(pos_features, neg_features, embeddings, edges, non_edges):
    pos_features_partial = np.abs(embeddings[edges[:, 0]] - embeddings[edges[:, 1]])
    neg_features_partial = np.abs(embeddings[non_edges[:, 0]] - embeddings[non_edges[:, 1]])
    return append_features(pos_features, neg_features, pos_features_partial, neg_features_partial)


def setup_filenames_and_folders(args, chromosome_folder):
    hyperparameters = ''
    if args.method != 'distance':
        hyperparameters = 'es{}'.format(args.emb_size)

    if args.method == 'node2vec':
        hyperparameters += '_nw{}_wl{}_p{}_q{}'.format(args.num_walks, args.walk_len, args.p, args.q)

    args.aggregators = '_'.join(args.aggregators)

    args.embedding = args.chromatin_network_name + '_' + hyperparameters

    os.makedirs('{}/results/{}/chr_{}'.format(args.data_root, args.dataset, chromosome_folder), exist_ok=True)
    os.makedirs('{}/results/{}/predictions/chr_{}'.format(args.data_root, args.dataset, chromosome_folder), exist_ok=True)
    if args.test:
        os.makedirs('{}/results/{}/test/chr_{}'.format(args.data_root, args.dataset, chromosome_folder), exist_ok=True)
        os.makedirs('{}/results/{}/test/predictions/chr_{}'.format(args.data_root, args.dataset, chromosome_folder), exist_ok=True)
    if args.method == 'topological':
        filename = '{}chr_{}/{}_{}_{}_{}_{}.pkl'.format('test/' if args.test else '', chromosome_folder, args.classifier,
                                                     args.method, args.chromatin_network_name, args.aggregators, args.times)
    else:
        filename = '{}chr_{}/{}_{}_{}_{}_{}_{}.pkl'.format('test/' if args.test else '', chromosome_folder, args.classifier,
                                                        args.method, args.embedding, args.aggregators, args.coexp_thr, args.times)
    return args, filename

def load_coexpression(args, chromatin_network_name, chrs):
    coexpression = np.load(
        '{}/{}/coexpression_networks/coexpression_chr_{}_{}.npy'.format(args.data_root, args.dataset, chrs, args.coexp_thr))

    chromatin_network = np.load('{}/{}/chromatin_networks/{}.npy'.format(args.data_root, args.dataset, chromatin_network_name))
    degrees = np.nansum(chromatin_network, axis=0)
    disconnected_nodes = np.ravel(np.argwhere(degrees == 0))

    print("N. disconnected nodes:", len(disconnected_nodes))
    if len(disconnected_nodes) > 0:
        coexpression[disconnected_nodes] = np.nan
        coexpression[:, disconnected_nodes] = np.nan
    return coexpression, disconnected_nodes

def get_edges(coexpression, n_eges_intra=None, inter_ratio=1.0):
    n_nodes = coexpression.shape[0]

    edges = np.array(np.argwhere(coexpression == 1))

    import random
    np.random.seed(42)
    random.seed(42)
    
    if n_eges_intra:
        if n_eges_intra > edges.shape[0]:
            n_edges_inter = edges.shape[0]
        else:
            n_edges_inter = int(n_eges_intra * inter_ratio)
        print('N. intra edges', n_eges_intra, '- N. inter edges ', edges.shape[0], '->',
              n_edges_inter)
        edges = edges[
            np.random.choice(edges.shape[0], n_edges_inter, replace=False)]

    # when make genome-wide intra-chrom prediction, could sample some edges as positive labels.
    # edges = edges[
    #         np.random.choice(edges.shape[0], int(edges.shape[0]*0.8), replace=False)]
    edges_nodes = np.unique(edges)

    non_nodes = np.setdiff1d(np.arange(n_nodes), edges_nodes)

    coexpression_neg = coexpression.copy()
    coexpression_neg[non_nodes, :] = np.nan
    coexpression_neg[:, non_nodes] = np.nan

    non_edges = np.array(np.argwhere(coexpression_neg == 0))
    non_edges = non_edges[
        np.random.choice(non_edges.shape[0], edges.shape[0], replace=False)]

    return edges, non_edges

def build_dataset(args, edges, non_edges, n_nodes):
    if args.method == 'topological':
        X = topological_features(args, edges, non_edges)
    elif args.method == 'ids':
        X = np.vstack((edges, non_edges))
    elif args.method == 'distance':
        X = distance_embedding(args.data_root, args.dataset, edges, non_edges)
    elif args.method.split('_')[0] == 'GNN':
        X = np.vstack((edges, non_edges))
    else:
        X = method_embedding(args, n_nodes, edges, non_edges)
    y = np.hstack((np.ones(edges.shape[0]), np.zeros(non_edges.shape[0])))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def link_prediction(args, X_train, y_train, X_test, y_test, filename, verbose=False):
    results = defaultdict(list)
    if args.test:
        results = evaluate_embedding(X_train, y_train, args.classifier, verbose=verbose, clf_params={'n_estimators': 100},
                                     X_test=X_test, y_test=y_test)
    else:
        for i in range(args.n_iter):
            results_iter = evaluate_embedding(X_train, y_train, args.classifier, verbose=verbose,
                                              clf_params={'n_estimators': 100}, cv_splits=args.cv_splits)
            for key in results_iter.keys():
                results[key].extend(results_iter[key])

    with open('{}/results/{}/{}'.format(args.data_root, args.dataset, filename), 'wb') as file_save:
        pickle.dump(results, file_save)

    print("Mean Accuracy: {:.3f} - Mean F1: {:.3f} - Mean Precision: {:.3f} - Mean Recall: {:.3f}".format(np.mean(results['acc']), np.mean(results['f1']), np.mean(results['precision']), np.mean(results['recall'])))
    print('')

def get_mask_intra(path, dataset):
    shapes = [np.load(
        '{}/{}/coexpression/coexpression_chr_{}_{}.npy'.format(path, dataset, i, i)).shape for i
              in
              range(1, 23)]

    mask = intra_mask(shapes, nans=True, values=np.ones)
    return mask

def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x


def betweenness_centrality_parallel(G, processes=None):
    """Parallel betweenness centrality  function"""
    p = Pool(processes=processes)
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(G.nodes(), int(G.order() / node_divisor)))
    num_chunks = len(node_chunks)
    bt_sc = p.starmap(
        nx.betweenness_centrality_source,
        zip([G] * num_chunks, [True] * num_chunks, [None] * num_chunks, node_chunks),
    )

    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c
