import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main(args):
    data_folder = '{}/{}/'.format(args.data_root, args.dataset)

    if args.chr_src != args.chr_tgt:
        raise NotImplementedError

    if args.chr_src and args.chr_tgt:
        chr_src = args.chr_src
        chr_tgt = args.chr_tgt
    else:
        chr_src = 'all'
        chr_tgt = 'all'

    if not os.path.exists('{}/{}/chr_sizes.npy'.format(args.data_root,args.dataset)) or \
            not os.path.exists(data_folder + 'coexpression/coexpression_chr_{}_{}.npy'.format(chr_src, chr_tgt)) or \
            args.force:

        rna_folder = data_folder + 'rna/'
        os.makedirs(rna_folder, exist_ok=True)
   
        df = pd.read_csv(data_folder + 'expression_raw.csv', index_col=0)
        df = df.dropna()
        df = df.sort_values(['Chromosome/scaffold name', 'Transcription start site (TSS)'])

        if args.chr_src and args.chr_tgt:
            df_chr_src = df[df['Chromosome/scaffold name'] == chr_src]
            df_chr_tgt = df[df['Chromosome/scaffold name'] == chr_tgt]
            #print('Computing co-expression between chromosome', chr_src, 'and chromosome', chr_tgt)
        else:
            df_chr_src = df
            df_chr_tgt = df
            print('Computing co-expression for all the chromosomes together')

        if chr_src == chr_tgt:
            df_chr_src.to_csv(rna_folder + 'expression_info_chr_{}.csv'.format(chr_src), )

        gene_exp_src = df_chr_src.iloc[:, 5:].to_numpy()
        gene_exp_tgt = df_chr_tgt.iloc[:, 5:].to_numpy()

        coexp = np.corrcoef(gene_exp_src, gene_exp_tgt)[:gene_exp_src.shape[0], -gene_exp_tgt.shape[0]:]
        #coexp = np.corrcoef(gene_exp_src)
        coexp[np.tril_indices_from(coexp, k=0)] = np.nan

        if args.chr_src == args.chr_tgt:
            if os.path.exists('{}/{}/chr_sizes.npy'.format(args.data_root, args.dataset)):
                chr_sizes = np.load('{}/{}/chr_sizes.npy'.format(args.data_root, args.dataset))
            else:
                chr_sizes = np.empty(23)

            chr_sizes[args.chr_src] = coexp.shape[0]
            np.save('{}/{}/chr_sizes.npy'.format(args.data_root, args.dataset), chr_sizes)

        if args.save_coexp:
            os.makedirs(data_folder + 'coexpression', exist_ok=True)

            np.save(data_folder + 'coexpression/coexpression_chr_{}_{}.npy'.format(chr_src, chr_tgt),
                    coexp)
    else:
        #print('Co-expression between chromosome', chr_src, 'and chromosome', chr_tgt, 'already computed. Skipped')
        if args.save_plot and not os.path.exists(
                '{}/plots/{}/coexpression/coexpression_chr_{}_{}.png'.format(args.data_root, args.dataset, chr_src, chr_tgt)):
            coexp = np.load(data_folder + 'coexpression/coexpression_chr_{}_{}.npy'.format(chr_src, chr_tgt))

    if args.save_plot and not os.path.exists('{}/plots/{}/coexpression/coexpression_chr_{}_{}.png'.format(args.data_root, args.dataset, chr_src, chr_tgt)):
        plt.imshow(1 - coexp, cmap='RdBu')
        os.makedirs('{}/plots/{}/coexpression'.format(args.data_root, args.dataset), exist_ok=True)

        plt.savefig('{}/plots/{}/coexpression/coexpression_chr_{}_{}.png'.format(args.data_root, args.dataset, chr_src, chr_tgt))
        # plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data-root', type=str, required=True, default='../../data')
    parser.add_argument('--dataset', type=str, default='adrenal_gland',
                        help='Name used to identify the dataset')
    parser.add_argument('--chr-src', type=int, default=None,
                        help='Source chromosome. If empty all the chromosomes are considered at once')
    parser.add_argument('--chr-tgt', type=int, default=None,
                        help='Target chromosome')
    parser.add_argument('--save-plot', default=True, action='store_true',
                        help='Save co-expression plot in results folder')
    parser.add_argument('--save-coexp', default=False, action='store_true',
                        help='Save co-expression numpy data in data/coexp folder')
    parser.add_argument('--force', default=False, action='store_true')

    args = parser.parse_args()
    main(args)

    if args.chr_src is None:
        for i in range(1, 23):
            args.chr_src = i
            args.chr_tgt = i
            main(args)
