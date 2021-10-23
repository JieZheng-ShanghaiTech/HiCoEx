import argparse
import os

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

from utils_network_building import chromatin_threshold, intra_mask, build_full_hic
from utils import block_matrix


def save_disconnected_nodes(hic, path, dataset, filename):
    degrees = np.sum(hic, axis=0)
    disconnected_nodes = np.ravel(np.argwhere(degrees == 0))
    os.makedirs('{}/{}/disconnected_nodes'.format(path, dataset), exist_ok=True)
    np.save('{}/{}/disconnected_nodes/{}.npy'.format(path, dataset, filename), disconnected_nodes)


# ToDo: make single function for common code
def single_chromosome(args):
    hic_name = '{}_{}_{}_{}'.format(args.type, args.chr_src, args.chr_tgt, args.resolution)

    data_folder = '{}/{}/chromatin_networks/'.format(args.data_root, args.dataset)
    filename = '{}_{}'.format(hic_name, args.perc_intra)

    if not os.path.exists(data_folder + filename + '.npy'):

        hic = np.load(
            '{}/{}/hic/{}.npy'.format(args.data_root, args.dataset, hic_name))

        print('Computing chromatin interaction network between chr.', args.chr_src, 'and chr.', args.chr_tgt)

        hics = [
            np.load('{}/{}/hic/{}_{}_{}_{}.npy'.format(args.data_root, args.dataset, args.type, i, i,
                                                                     args.resolution)) for i in range(1, 23)]

        # ToDo: remove or improve implementation for generating inter-chromosomal coexpression
        is_intra = (args.chr_src == args.chr_tgt)
        if is_intra and args.perc_intra is not None:
            threshold = chromatin_threshold(block_matrix(hics), percentile_intra=args.perc_intra)
        elif not is_intra and args.perc_inter is not None:
            raise NotImplementedError
        else:
            # ToDo: give useful error text
            raise ValueError()

        hic[hic <= threshold] = 0
        hic[hic > 0] = 1

        print('N. edges after thresholding', (hic == 1).sum())

        save_disconnected_nodes(hic, args.data_root, args.dataset, filename)

        if args.save_matrix:
            os.makedirs(data_folder, exist_ok=True)
            np.save(
                data_folder + filename + '.npy', hic)
    else:
        print('Chromatin network for chr.', args.chr_src, 'already computed. Skipped')
        if args.save_plot:
            hic = np.load(data_folder + filename + '.npy')

    if args.save_plot:
        plt.imshow(np.log1p(hic), cmap="Reds")
        os.makedirs('{}/plots/{}/chromatin_networks'.format(args.data_root, args.dataset), exist_ok=True)
        plt.savefig('{}/plots/{}/chromatin_networks/{}.png'.format(args.data_root, args.dataset, filename))


def multi_chromosome(args):
    hic_name = '{}_all_{}'.format(args.type, args.resolution)

    data_folder = '{}/{}/chromatin_networks/'.format(args.data_root, args.dataset)

    if args.perc_inter is not None:
        filename = '{}_{}_{}'.format(hic_name, args.perc_intra, args.perc_inter)
    else:
        filename = '{}_{}'.format(hic_name, args.perc_intra)

    if not os.path.exists(data_folder + filename + '.npy'):

        hics = [
            np.load(
                '{}/{}/hic/{}_{}_{}_{}.npy'.format(args.data_root, args.dataset, args.type, i, i,
                                                                 args.resolution)) for i in range(1, 23)]

        if args.perc_inter is not None:
            shapes = list(map(lambda x: x.shape, hics))

            mask = intra_mask(shapes)
            hic_full = build_full_hic(args.data_root, args.dataset, args.type, args.resolution)
            threshold_inter = chromatin_threshold(hic_full, mask=mask, percentile_inter=args.perc_inter)

            hic_inter = hic_full * np.logical_not(mask)
            hic_inter[hic_inter < threshold_inter] = 0
            hic_inter[hic_inter > 0] = 1

            hic_intra = hic_full * mask

            hic_name += '_{}_all_{}'.format( args.type_inter, args.resolution)
        else:
            hic_intra = block_matrix(hics)
            # np.save(
            #     '{}/{}/hic/{}.npy'.format(args.data_root, args.dataset, filename), hic_intra)
            hic_inter = np.zeros(hic_intra.shape)
        # import ipdb
        # ipdb.set_trace()
        threshold_intra = chromatin_threshold(hic_intra, percentile_intra=args.perc_intra)
        print(threshold_intra)

        hic_intra[hic_intra < threshold_intra] = 0
        hic_intra[hic_intra > 0] = 1

        print(np.nansum(hic_intra), np.nansum(hic_inter> 0))

        hic_thr = hic_intra + hic_inter



        save_disconnected_nodes(hic_thr, args.data_root, args.dataset, filename)

        if args.save_matrix:

            os.makedirs(data_folder, exist_ok=True)
            np.save(
                data_folder + filename + '.npy'.format(filename), hic_thr)
    else:
        print('Chromatin network for genome already computed. Skipped')

        if args.save_plot:
            hic_thr = np.load(data_folder + filename + '.npy'.format(filename))

    if args.save_plot:
        plt.figure(figsize=(7, 7), dpi=600)
        plt.imshow(np.log1p(hic_thr), cmap='Oranges')
        os.makedirs('{}/plots/{}/chromatin_networks'.format(args.data_root, args.dataset), exist_ok=True)
        plt.savefig(
            '{}/plots/{}/chromatin_networks/{}.png'.format(args.data_root, args.dataset, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True, default='../../data')
    parser.add_argument('--dataset', type=str, default='adrenal')
    parser.add_argument('--chr-src', type=int, default=None)
    parser.add_argument('--chr-tgt', type=int, default=None)

    parser.add_argument('--type', type=str, choices=['observed', 'oe', 'primary_observed_ICE'], default='observed')
    parser.add_argument('--resolution', type=int, default=40000)

    parser.add_argument('--type-inter', type=str, choices=['observed', 'oe'], default='observed')
    parser.add_argument('--resolution-inter', type=int, default=40000)

    parser.add_argument('--perc-intra', type=float, default=80)
    parser.add_argument('--perc-inter', type=float, default=None)
    parser.add_argument('--single-chrom', default=False, action='store_true')
    parser.add_argument('--save-plot', default=False, action='store_true')
    parser.add_argument('--save-matrix', default=False, action='store_true')
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
