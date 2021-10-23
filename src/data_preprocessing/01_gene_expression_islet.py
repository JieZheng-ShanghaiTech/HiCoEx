import argparse
import os

import numpy as np
import pandas as pd


def main(args, tcga, tag):
    dataset_path = '../../data/pancreatic_islet_{}'.format(tag)
    if not os.path.exists(dataset_path + '/expression_raw.csv'):
        os.makedirs(dataset_path) 

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
    parser.add_argument('--dataset', type=str, default='pancreatic_islet',
                        help='Name that will be used to identify the dataset')
    parser.add_argument('--gene-info', type=str, default='../../data/GRCh37_p13_gene_info.txt',
                        help='Path of the txt file containing the association gene name - TSS')
    args = parser.parse_args()

    tmm = pd.read_excel(args.input)
    tmm = tmm.rename(columns={'id': 'sample'})
    tmm = tmm.set_index('sample')
    tmm = np.log1p(tmm)

    # import ipdb
    # ipdb.set_trace()
    hba1c = [5.8, None, 5.4, None, None, 5.5, None, 4.3, None, None, 
        6.9, None, 5.4, 5.8, 5.5, 4.6, 6.1, 5.3, 4.5, 4.3, None, 
        5.4, 5.5, 6.2, 5.8, 6, 5.9, 7, 5.6, 5.6, 8.6, 5.4, 5.3, 
        6.2, 5, 5.2, 6, 5.3, 5.6, None, 6.8, 6.8, 5.5, 5.3, 8, 
        4.6, 6.2, None, 5.6, 6, 5.5, 6.2, 6.4, 7, 10, 5.1, 5.3, 
        5.4, 5.3, 6, 5.8, 5.4, 5.7, 6.2, 5.2, 5.7, 5.2, 5.4, 5.6, 
        5.6, 5.8, 5.7, 5.6, 5.7, 6.1, 6.1, 7.3, None, None, 5.7, 
        5.5, 7.2, 6.4, 6.7, 5.9, 5.6, 6, 5.6, 5.3]
    hba1c = np.array(hba1c)
    hba1c[hba1c==None] = 0.
    sub0 = (hba1c>=4) & (hba1c < 6.4)
    sub1 = hba1c>=6.5

    tmm_sub0 = tmm.iloc[:,sub0].reset_index()
    tmm_sub1 = tmm.iloc[:,sub1].reset_index()
    
    main(args, tmm_sub0, 'healthy')
    main(args, tmm_sub1, 'diabetic')
