import argparse
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import set_n_threads
from utils_link_prediction import generate_embedding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True, default='../../data')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--chromatin-network', type=str, required=True)
    parser.add_argument('--emb-size', type=int, default=16)
    parser.add_argument('--n-jobs', type=int, default=10)
    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--method', default='SVD', type=str, choices=['Laplacian', 'SVD', 'HOPE'])
    parser.add_argument('--weighted', type=str, default='False')
    parser.add_argument('--save-emb', default=False, action='store_true')
    parser.add_argument('--task', type=str, default='link-prediction', choices=['none', 'link-prediction'])
    args = parser.parse_args()

    set_n_threads(args.n_jobs)

    args.folder = 'chromatin_networks'
    args.name = args.chromatin_network

    interactions_path = '{}/{}/chromatin_networks/{}.npy'.format(args.data_root, args.dataset, args.name)

    emb_path = '{}_es{}'.format(args.chromatin_network, args.emb_size)


    if not os.path.exists(
            '{}/{}/embeddings/{}/{}.npy'.format(args.data_root, args.dataset, args.method.lower(), emb_path)) or args.force:
        command = 'bionev --input {}/{}/chromatin_networks/{}.edgelist '.format(args.data_root, args.dataset,
                                                                                        args.chromatin_network) + \
                  '--output {}/{}/embeddings/{}/{}.txt '.format(args.data_root, args.dataset, args.method.lower(), emb_path) + \
                  '--method {} --task {} '.format(args.method, args.task) + \
                  '--dimensions {} '.format(args.emb_size) + \
                  '--weighted True' if args.weighted else ''

        generate_embedding(args, emb_path, interactions_path, command)
    else:
        print('Embeddings already computed for {}. Skipped.'.format('{}/{}/embeddings/{}/{}.npy'.format(args.data_root, args.dataset, args.method.lower(), emb_path)))
