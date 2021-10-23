import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from utils_network_building import coexpression_threshold, intra_mask


def single_chromosome(args):
    data_folder = '{}/{}/coexpression_networks'.format(args.data_root, args.dataset)
    os.makedirs(data_folder, exist_ok=True)

    filename_plot = '{}/plots/{}/coexpression_networks/coexpression_chr_{}_{}_{}{}.png'.format(
        args.data_root, args.dataset, args.chr_src,
        args.chr_tgt,
        args.perc_intra,
        '_abs' if args.abs else '')

    if not os.path.exists(
            data_folder + '/coexpression_chr_{}_{}_{}{}.npy'.format(args.chr_src, args.chr_tgt, args.perc_intra,
                                                                    '_abs' if args.abs else '')):
        coexpression = np.load(
            '{}/{}/coexpression/coexpression_chr_{}_{}.npy'.format(args.data_root, args.dataset, args.chr_src, args.chr_tgt))

        print('Computing co-expression network between chromosome', args.chr_src, 'and chromosome', args.chr_tgt)

        if args.abs:
            coexpression = np.abs(coexpression)

        # ToDo: implement that if single chromosome computes the threshold not shared
        # ToDo: remove or improve implementation for generating inter-chromosomal coexpression
        is_intra = (args.chr_src == args.chr_tgt)
        if is_intra and args.perc_intra is not None:
            threshold = coexpression_threshold(args.data_root, args.dataset, percentile_intra=args.perc_intra)
        elif not is_intra and args.perc_inter is not None:
            threshold = coexpression_threshold(args.data_root, args.dataset, percentile_inter=args.perc_inter)
        else:
            raise ValueError('Either one parameter between --percentile and --threshold must be passed.')

        threshold = np.round(threshold, 2)

        coexpression[coexpression < threshold] = 0
        coexpression[coexpression >= threshold] = 1

        print('N. edges after thresholding', (coexpression == 1).sum())

        if args.save_matrix:
            np.save(
                data_folder + '/coexpression_chr_{}_{}_{}{}.npy'.format(args.chr_src,
                                                                        args.chr_tgt,
                                                                        args.perc_intra,
                                                                        '_abs' if args.abs else ''),
                coexpression)
    else:
        print('Coexpression network for chromosome', args.chr_src, 'already computed. Skipped')
        if args.save_plot and not os.path.exists(filename_plot):
            coexpression = np.load(data_folder + '/coexpression_chr_{}_{}_{}{}.npy'.format(args.chr_src, args.chr_tgt,
                                                                                           args.perc_intra,
                                                                                           '_abs' if args.abs else ''))
    if args.save_plot and not os.path.exists(filename_plot):
        os.makedirs('{}/plots/{}/coexpression_networks/'.format(args.data_root, args.dataset), exist_ok=True)
        plt.imshow(coexpression, cmap='Oranges')
        plt.savefig(filename_plot)
        plt.close()


def multi_chromosome(args):
    data_folder = '{}/{}/coexpression_networks'.format(args.data_root, args.dataset)
    if args.perc_inter is not None:
        filename = 'coexpression_chr_all_{}_{}'.format(args.perc_intra, args.perc_inter)
    else:
        filename = 'coexpression_chr_all_{}'.format(args.perc_intra)

    if not os.path.exists(data_folder + '/' + filename + '.npy'.format(filename)) or args.force:
        coexpression_full = np.load('{}/{}/coexpression/coexpression_chr_all_all.npy'.format(args.data_root, args.dataset))

        shapes = [np.load('{}/{}/coexpression/coexpression_chr_{}_{}.npy'.format(args.data_root, args.dataset, i, i)).shape for
                  i in
                  range(1, 23)]

        mask = intra_mask(shapes)

        if args.perc_inter is not None:
            threshold_inter = coexpression_threshold(args.data_root, args.dataset, percentile_inter=args.perc_inter)
            coexpression_inter = coexpression_full * np.logical_not(mask)
            coexpression_inter[coexpression_inter < threshold_inter] = 0
            coexpression_inter[coexpression_inter > 0] = 1
        else:
            coexpression_inter = np.ones(coexpression_full.shape)
            coexpression_inter *= np.logical_not(mask)
            coexpression_inter[coexpression_inter == 1] = np.nan

        threshold_intra = coexpression_threshold(args.data_root, args.dataset, percentile_intra=args.perc_intra)
        coexpression_intra = coexpression_full * mask
        coexpression_intra[coexpression_intra < threshold_intra] = 0
        coexpression_intra[coexpression_intra > 0] = 1
        
        print('N. intra-chromosomal interactions:', np.nansum(coexpression_intra))
        print('N. inter-chromosomal interactions:', np.nansum(coexpression_inter))

        coexpression_thr = coexpression_intra + coexpression_inter

        if args.save_matrix:
            os.makedirs(data_folder, exist_ok=True)
            np.save(data_folder + '/' + filename + '.npy'.format(filename), coexpression_thr)

        if args.save_plot:
            os.makedirs('{}/plots/{}/coexpression_networks/'.format(args.data_root, args.dataset), exist_ok=True)
            plt.figure(figsize=(7, 7), dpi=600)
            plt.imshow(coexpression_thr, cmap='Oranges')
            plt.savefig(
                '{}/plots/{}/coexpression_networks/{}.png'.format(args.data_root, args.dataset, filename))
            plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True, default='../../data')
    parser.add_argument('--dataset', type=str, default='prostate')
    parser.add_argument('--chr-src', type=int, default=None)
    parser.add_argument('--chr-tgt', type=int, default=None)
    parser.add_argument('--perc-intra', type=float, default=None)  # , required=True)
    parser.add_argument('--perc-inter', type=float, default=None)
    parser.add_argument('--single-chrom', default=False, action='store_true')
    parser.add_argument('--abs', default=False, action='store_true')
    parser.add_argument('--save-plot', default=False, action='store_true')
    parser.add_argument('--save-matrix', default=False, action='store_true')
    parser.add_argument('--force', default=False, action='store_true')
    args = parser.parse_args()

    if args.chr_src and args.chr_tgt:
        single_chromosome(args)
    else:
        if not args.single_chrom:
            multi_chromosome(args)

        if args.perc_inter is None:
            for i in range(1, 23):
                args.chr_src = i
                args.chr_tgt = i
                single_chromosome(args)
