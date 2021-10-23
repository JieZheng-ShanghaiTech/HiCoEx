import numpy as np
import random
import torch

import argparse
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from utils_link_prediction import *
from utils import set_n_threads, set_gpu
from train_GNN import train_main 

import ipdb
def main(args):
    if args.chr_src is None:
        raise ValueError()

    if args.chr_tgt is None:
        args.chr_tgt = args.chr_src

    if args.chromatin_network_name is None:
        args.chromatin_network_name = '{}_{}_{}_{}_{}'.format(args.type, args.chr_src, args.chr_tgt, args.bin_size, args.hic_threshold)

    args, filename = setup_filenames_and_folders(args, args.chr_src)

    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists('{}/results/{}/{}'.format(args.data_root, args.dataset, filename)) or args.force:
        print('Prediction of co-expression links from chr. {} to {} using {} embeddings.'.format(args.chr_src, args.chr_tgt, args.method))
        coexpression, disconnected_nodes = load_coexpression(args, args.chromatin_network_name, '{}_{}'.format(args.chr_src, args.chr_tgt))

        edges, non_edges = get_edges(coexpression)

        X_train, X_test, y_train, y_test = build_dataset(args, edges, non_edges, coexpression.shape[0])
        print('Method: {}, classifier: {}, seed: {}'.format(args.method, args.classifier, args.seed))
        if args.method[:3] != 'GNN':
            if args.chr_src == 1:
                print('Training on {} dataset'.format(args.dataset))
                print('{} training samples, {} testing samples'.format(len(X_train), len(X_test)))
                
            link_prediction(args, X_train, y_train, X_test, y_test, filename)
        else:
            emb = train_main(args, X_train, y_train, X_test, y_test, filename)
            X = np.vstack((X_train, X_test))
            y = np.hstack((y_train, y_test))
            name = args.chromatin_network_name
            
            emb_file = '{}_es{}_nl{}_nhds{}_clf_{}'.format(name, args.emb_size,
                                                        args.n_layers, args.num_heads, args.classifier)
            if args.save_emb:
                emb_path = '{}/{}/embeddings/{}_{}'.format(args.data_root, args.dataset, args.classifier, args.method)
                os.makedirs(emb_path, exist_ok=True)
                np.save('{}/{}.npy'.format(emb_path, emb_file), emb.cpu().numpy())
        
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
    parser.add_argument('--chr-src', type=int, default=1)
    parser.add_argument('--chr-tgt', type=int, default=1)
    parser.add_argument('--n-iter', type=int, default=1)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--method', type=str, default='node2vec',
                        choices=['random', 'distance', 'topological', 'svd', 'node2vec', 
                                'GNN_GCN', 'GNN_HiCoEx', 'GNN_GCN_pyg', 'GNN_HiCoEx_pyg'])

    parser.add_argument('--type', type=str)
    parser.add_argument('--bin-size', type=int)
    parser.add_argument('--hic-threshold', type=str)
    parser.add_argument('--chromatin-network-name', type=str)

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

    # GNN_* params
    parser.add_argument('--training', default=False, action='store_true')
    parser.add_argument('--times', type=int, default=None)
    parser.add_argument('--init_lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--lr-reduce-factor', type=float, default=0.5)
    parser.add_argument('--epoches', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=1)
    parser.add_argument('--out-dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--load-ckpt', default=False, action='store_true')
    parser.add_argument('--save-emb', default=False, action='store_true')

    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--test', default=True, action='store_true')
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

    main(args)
