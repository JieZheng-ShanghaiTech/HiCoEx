import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sps
import matplotlib.pyplot as plt
 

def main(args):
    print('Loading using juicer')

    chromosomes = range(1, 23) if args.chromosomes is None else args.chromosomes

    dataset_path = '../../data/{}/hic_raw/'.format(args.dataset)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)


    for i, chr_source in enumerate(chromosomes):

        if args.inter:
            chromosomes_target = chromosomes[i:]
        else:
            chromosomes_target = [chr_source]

        for chr_target in chromosomes_target:
            print('Downloading interactions between chr.', chr_source, 'and chr.', chr_target)
            output_hic = 'hic_raw_{}_{}_{}.npz'.format(chr_source, chr_target, args.window)
            output_path = os.path.join(dataset_path, output_hic)

            if not os.path.exists(output_path):
                
                # import ipdb
                # ipdb.set_trace()
                output_original = 'hic_raw_{}_{}_{}.txt'.format(chr_source, chr_target, args.resolution)
                original_path = os.path.join(dataset_path, output_original)
                os.system('java -jar {} '.format(args.juicer_path) +
                          'dump observed NONE {} '.format(args.input) +
                          '{} {} '.format(chr_source, chr_target) +
                          'BP {} '.format(args.resolution) +
                          '{}'.format(original_path))

                hic = pd.read_csv(original_path, delim_whitespace=True, header=None)
                contact_matrix = sps.csr_matrix(
                    (hic.iloc[:, 2], (hic.iloc[:, 0] // args.resolution, hic.iloc[:, 1] // args.resolution)))

                if args.window > args.resolution:
                    contact_matrix_agg = np.add.reduceat(contact_matrix.toarray(),
                                                         np.arange(0, contact_matrix.shape[0], args.window // args.resolution),
                                                         axis=0)
                    contact_matrix_agg = np.add.reduceat(contact_matrix_agg,
                                                         np.arange(0, contact_matrix.shape[1], args.window // args.resolution),
                                                         axis=1)
                    contact_matrix = sps.csr_matrix(contact_matrix_agg)

                # import ipdb
                # ipdb.set_trace()
                sps.save_npz(output_path, contact_matrix)
                os.remove(original_path)
            else:
                print('File already existing. Skip.')
                if args.save_plot:
                    contact_matrix = sps.load_npz(output_path)

            if args.save_plot:
                plot_path = '../../data/plots/{}/hic_raw'.format(args.dataset)
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path, exist_ok=True)
                plt.figure(dpi=200)
                plt.imshow(np.log1p(contact_matrix.toarray()*10), cmap='Reds')
                plt.savefig(os.path.join(plot_path, 'hic_raw_{}_{}_{}.png'.format(chr_source, chr_target, args.window)))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True, help='Link of the Hi-C data hosted on Juicer')
    parser.add_argument('--juicer-path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=10000,
                        help='Resolution of the Hi-C data.')
    parser.add_argument('--window', type=int, default=40000,
                        help='Resolution of the Hi-C data.')
    parser.add_argument('--chromosomes', nargs='*', default=None,
                        help='List of chromosomes for which to extract the Hi-C data. If empty all the non-sexual chromosomes data will be extracted.')
    parser.add_argument('--inter', default=False, action='store_true',
                        help='Extract also interchromosomal interactions')
    parser.add_argument('--save-plot', default=False, action='store_true')

    args = parser.parse_args()

    if args.window % args.resolution != 0:
        raise ValueError('window must be a multiple of the resolution')

    main(args)
