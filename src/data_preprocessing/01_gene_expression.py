import argparse
import os

import pandas as pd


def main(args):
    dataset_path = '../../data/{}'.format(args.dataset)
    if not os.path.exists(dataset_path + '/expression_raw.csv'):
        tcga = pd.read_csv(args.input, delimiter='\t')
        print('Gene expression data loaded:', tcga.shape[0], 'genes and', tcga.shape[1] - 1, 'samples')

        low_expression_genes = (tcga == 0).sum(axis=1) <= tcga.shape[1] * 0.2

        print(tcga.shape[0] - low_expression_genes.sum(), 'genes out of', tcga.shape[0],
              'have more that 80% of samples with 0 expression. Removed')
        tcga = tcga[low_expression_genes]

        gene_info = pd.read_csv(args.gene_info, delimiter='\t')

        print('Merging gene expression data with gene information from Ensembl hg19')
        tcga = gene_info.merge(tcga, right_on='sample', left_on='Gene name', )
        tcga = tcga.drop('sample', axis=1)

        print('Removing duplicated gene entries and keeping the ones with the outermost TSS')
        tcga_pos = tcga[tcga['Strand'] == 1]
        tcga_neg = tcga[tcga['Strand'] == -1]
        tcga_pos = tcga_pos.groupby(['Gene name']).min()
        tcga_neg = tcga_neg.groupby(['Gene name']).max()

        tcga = pd.concat([tcga_neg, tcga_pos])
        tcga = tcga.groupby(['Gene name']).max()

        print('Final gene expression data with', tcga.shape[0], 'genes')

        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)

        print('Saving in', dataset_path + '/expression_raw.csv')
        tcga.to_csv(dataset_path + '/expression_raw.csv')
    else:
        print('Expression already computed. Skipped.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True,
                        help='Gene expression input file path downloaded from Xena Browser')
    parser.add_argument('--dataset', type=str, default='breast_normal',
                        help='Name that will be used to identify the dataset')
    parser.add_argument('--gene-info', type=str, default='../../data/GRCh37_p13_gene_info.txt',
                        help='Path of the txt file containing the association gene name - TSS')
    args = parser.parse_args()

    main(args)
