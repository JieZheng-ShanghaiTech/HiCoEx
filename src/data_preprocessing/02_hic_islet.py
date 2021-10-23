import os
import argparse

import numpy as np
import cooler
import scipy.sparse as sps
from iced import normalization

def main(args):
    chromosomes = range(1, 23) if args.chromosomes is None else args.chromosomes

    dataset_path = '../../data/pancreatic_islet/hic_raw'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)

    input_path = '{}::resolutions/{}'.format(args.input, args.resolution)
    obj = cooler.Cooler(input_path)

    for i, chr_source in enumerate(chromosomes):

        if args.inter:
            chromosomes_target = chromosomes[i:]
        else:
            chromosomes_target = [chr_source]

        for chr_target in chromosomes_target:
            print('Extracting interactions between chr.', chr_source, 'and chr.', chr_target)
            output_hic = 'hic_raw_{}_{}_{}.npz'.format(chr_source, chr_target, args.window)
            output_path = os.path.join(dataset_path, output_hic)

            if not os.path.exists(output_path):
                
                # import ipdb
                # ipdb.set_trace()
                contact_matrix = obj.matrix(balance=False, sparse=True).fetch(str(chr_source), str(chr_target))
                
                contact_matrix = contact_matrix.toarray()
                contact_matrix = normalization.ICE_normalization(contact_matrix)
                contact_matrix_sparse = sps.csr_matrix(contact_matrix)
                sps.save_npz(output_path, contact_matrix_sparse)
            else:
                print('File already existing. Skip.')
                

            if args.save_plot:
                contact_matrix = sps.load_npz(output_path)
                plot_path = '../../data/plots/{}/hic_raw'.format(args.dataset)
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path, exist_ok=True)
                plt.figure(dpi=200)
                plt.imshow(np.log1p(contact_matrix.toarray()*10), cmap='Reds')
                plt.savefig(os.path.join(plot_path, 'hic_raw_{}_{}_{}.png'.format(chr_source, chr_target, args.window)))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--resolution', type=int, required=True, help='Resolution of the Hi-C data.')
    parser.add_argument('--window', type=int, default=40000, help='Resolution of the Hi-C data.')
    parser.add_argument('--chromosomes', nargs='*', default=None,
                        help='List of chromosomes for which to extract the Hi-C data. If empty all the non-sexual chromosomes data will be extracted.')
    parser.add_argument('--inter', default=False, action='store_true',
                        help='Extract also interchromosomal interactions')
    parser.add_argument('--save-plot', default=False, action='store_true')

    
    args = parser.parse_args()

    main(args)