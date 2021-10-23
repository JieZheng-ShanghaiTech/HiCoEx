import os
import argparse

import numpy as np
from iced import normalization
import scipy.sparse as sps

def main(args):
    chromosomes = range(1, 23) if args.chromosomes is None else args.chromosomes

    dataset_path = '../../data/{}/hic_raw'.format(args.dataset)
    
    file_list = os.listdir(dataset_path)
    chr_list = [f.split('.')[-2] for f in file_list]

    for i in chromosomes:
        print('Chromosome ', i)

        input_path = file_list[chr_list.index('chr'+str(i))]
        original_path = os.path.join(dataset_path, input_path)
        # import ipdb
        # ipdb.set_trace()
        contact_matrix = np.genfromtxt(original_path, delimiter='\t')
        contact_matrix = normalization.ICE_normalization(contact_matrix)
        contact_matrix_sparse = sps.csr_matrix(contact_matrix)

        sps.save_npz(dataset_path + '/hic_raw_{}_{}_{}.npz'.format(i, i, args.resolution),
                     contact_matrix_sparse)
        os.remove(original_path)
        
        if args.save_plot:
            plot_path = '../../data/plots/{}/hic_raw'.format(args.dataset)
            if not os.path.exists(plot_path):
                os.makedirs(plot_path, exist_ok=True)
            plt.figure(dpi=200)
            plt.imshow(np.log1p(contact_matrix.toarray()*10), cmap='Reds')
            plt.savefig(os.path.join(plot_path, 'hic_raw_{}_{}_{}.png'.format(chr_source, chr_target, args.window)))

    print('Hi-C data saved in sparse format in', dataset_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--resolution', type=int, required=True, help='Resolution of the Hi-C data.')
    parser.add_argument('--chromosomes', nargs='*', default=None,
                        help='List of chromosomes for which to normalize the Hi-C data. If empty all the non-sexual chromosomes data will be normalized.')
    parser.add_argument('--save-plot', default=False, action='store_true')

    args = parser.parse_args()

    main(args)