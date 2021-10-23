import argparse
import warnings
import random
import torch

warnings.simplefilter(action='ignore', category=FutureWarning)

from utils_link_prediction import *
from utils import set_n_threads, set_gpu
import matplotlib.pyplot as plt

from train_GNN import train_main 

def main(args):
    args.chromatin_network_name = '{}_all_{}_{}'.format(args.type,args.bin_size, args.hic_threshold)

    args, filename = setup_filenames_and_folders(args, 'all')

    edges = None
    non_edges = None
    offset = 0
    edges_cut, non_edges_cut = 0, 0
    if not os.path.exists('{}/results/{}/{}'.format(args.data_root, args.dataset, filename)) or args.force:
        print('Prediction of all the intra-chromosomal co-expression links using {} embeddings.'.format(args.method))
        for i in range(1, 23):
            chromatin_network_chr_name = '{}_{}_{}_{}_{}'.format(args.type, i, i,
                                                                       args.bin_size,
                                                                       args.hic_threshold)

            coexpression, disconnected_nodes = load_coexpression(args, chromatin_network_chr_name, '{}_{}'.format(i, i))

            edges_intra, non_edges_intra = get_edges(coexpression)

            edges_intra += offset
            non_edges_intra += offset

            if edges is None and non_edges is None:
                edges = edges_intra
                non_edges = non_edges_intra
            else:
                edges = np.vstack((edges, edges_intra))
                non_edges = np.vstack((non_edges, non_edges_intra))

            offset += coexpression.shape[0]
            if i > 18:
                edges_cut += len(edges_intra)
                non_edges_cut += len(non_edges_intra)

        X_train, X_test, y_train, y_test = build_dataset(args, edges[:-edges_cut], non_edges[:-non_edges_cut], offset)
        if args.method[:3] != 'GNN':
            print('Training on {} dataset'.format(args.dataset))
            print('{} training samples, {} testing samples'.format(len(X_train), len(X_test)))
            print('Method: {}, classifier: {}, seed: {}'.format(args.method, args.classifier, args.seed))
            link_prediction(args, X_train, y_train, X_test, y_test, filename, verbose=True)
        else:
            emb = train_main(args, X_train, y_train, X_test, y_test, filename)
            emb[disconnected_nodes, :] = np.nan

            name = args.chromatin_network_name
            emb_path = '{}_es{}_nl{}_nhds{}_clf_{}'.format(name, args.emb_size,
                                                        args.n_layers, args.num_heads, args.classifier)
            if args.save_emb:
                os.makedirs('{}/{}/embeddings/{}'.format(args.data_root, args.dataset, args.method), exist_ok=True)
                np.save('{}/{}/embeddings/{}/{}.npy'.format(args.data_root, args.dataset, args.method, emb_path), emb)
    else:
        print('Result already computed for {}. Skipped.'.format(filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True, default='../../data')
    parser.add_argument('--node-feature', type=str, default='random', 
                        choices=['random', 'one-hot', 'biological', 'pre-trained'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nfeat-path', type=str, default=None, 
                        help='require when node feature is biological or pre-trained')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--chr-src', type=int)
    parser.add_argument('--n-iter', type=int, default=1)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--method', type=str, default='node2vec',
                        choices=['random', 'distance', 'topological', 'svd', 'node2vec', 
                                'GNN_GCN', 'GNN_HiCoEx', 'GNN_GCN_pyg', 'GNN_HiCoEx_pyg'])

    parser.add_argument('--type', type=str, default='observed')
    parser.add_argument('--hic-threshold', type=str, required=True)
    parser.add_argument('--bin-size', type=int)
    parser.add_argument('--verbose', default=False, action='store_true')
    
    parser.add_argument('--aggregators', nargs='*', default=['hadamard'], choices=['hadamard', 'avg', 'l1'])
    parser.add_argument('--classifier', default='rf', choices=['mlp', 'direct', 'rf', 'random'])
    parser.add_argument('--coexp-thr', type=str, default=None, required=True)
    parser.add_argument('--save-predictions', default=True, action='store_true')
    parser.add_argument('--emb-size', type=int, default=16)

    # Topological measures params
    parser.add_argument('--edge-features', default=True, action='store_true')

    # Node2vec params
    parser.add_argument('--num-walks', type=int, default=10)
    parser.add_argument('--walk-len', type=int, default=80)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--q', type=float, default=1.0)
    parser.add_argument('--window', type=int, default=10)

    # Todo: add GNN_* params
    parser.add_argument('--training', default=False, action='store_true')
    parser.add_argument('--times', type=int, default=1)
    parser.add_argument('--init_lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--lr-reduce-factor', type=float, default=0.5)
    parser.add_argument('--epoches', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=1)
    parser.add_argument('--out-dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num-heads', type=int, default=1)
    parser.add_argument('--gnn-classifier', default='rf', choices=['mlp', 'lr', 'svm', 'rf', 'random', 'direct'])
    parser.add_argument('--load-ckpt', default=False, action='store_true')
    parser.add_argument('--save-emb', default=False, action='store_true')

    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--n-jobs', type=int, default=10)
    parser.add_argument('--wandb', default=False, action='store_false')
    parser.add_argument('--project', default='parameter-importance', type=str)
    args = parser.parse_args()

    # ipdb.set_trace()
    device = set_gpu(args.gpu, args.gpu_id)
    args.device = device
    set_n_threads(args.n_jobs)

    if args.chr_src is None:
        args.chr_src = 'all'

    main(args)



    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, required=True)
    # parser.add_argument('--data-root', type=str, required=True, default='../../data')
    # parser.add_argument('--n-iter', type=int, default=1)
    # parser.add_argument('--cv-splits', type=int, default=5)
    # parser.add_argument('--method', type=str, default='node2vec',
    #                     choices=['random', 'distance', 'topological', 'svd', 'node2vec'])

    # parser.add_argument('--type', type=str, default='observed')
    # parser.add_argument('--bin-size', type=int, default=40000)
    # parser.add_argument('--hic-threshold', type=str, required=True)

    # parser.add_argument('--edge-features', default=True, action='store_true')
    # parser.add_argument('--aggregators', nargs='*', default=['hadamard'])
    # parser.add_argument('--classifier', default='rf', choices=['mlp', 'lr', 'svm', 'mlp_2', 'rf'])
    # parser.add_argument('--coexp-thr', type=str, required=True)
    # parser.add_argument('--save-predictions', default=False, action='store_true')
    # parser.add_argument('--emb-size', type=int, default=16)
    # parser.add_argument('--force', default=False, action='store_true')
    # parser.add_argument('--test', default=False, action='store_true')
    

    # # Node2vec params
    # parser.add_argument('--num-walks', type=int, default=10)
    # parser.add_argument('--walk-len', type=int, default=80)
    # parser.add_argument('--p', type=float, default=1.0)
    # parser.add_argument('--q', type=float, default=1.0)
    # parser.add_argument('--window', type=int, default=10)

    # parser.add_argument('--gpu', default=False, action='store_true')
    # parser.add_argument('--n-jobs', type=int, default=10)
    # parser.add_argument('--wandb', default=True, action='store_true')
    # parser.add_argument('--project', type=str, default='n2v_hic_tuning')

    # args = parser.parse_args()

    # seed = 42
    # np.random.seed(seed)

    # set_gpu(args.gpu)
    # set_n_threads(args.n_jobs)

    # main(args)
