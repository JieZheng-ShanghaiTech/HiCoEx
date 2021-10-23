import os

import matplotlib.pyplot as plt
import numpy as np

from utils import intra_mask, block_matrix


def coexpression_threshold(path, dataset, percentile_intra=None, percentile_inter=None):
    if not percentile_intra and not percentile_inter:
        raise ValueError(
            'Either one parameter between percentile_intra and percentile_inter must be different from zero.')

    coexpression_full = np.load('{}/{}/coexpression/coexpression_chr_all_all.npy'.format(path, dataset))

    try:
        shapes = [np.load('{}/{}/coexpression/coexpression_chr_{}_{}.npy'.format(path, dataset, i, i)).shape for i in
                  range(1, 23)]
    except FileNotFoundError:
        raise ValueError('You need to compute the co-expression for all the chromosomes first.')

    mask = intra_mask(shapes)

    if percentile_intra:
        coexpression_intra = coexpression_full * mask
        coexpression_intra[coexpression_intra == 0] = np.nan
        threshold_intra = np.round(np.nanpercentile(coexpression_intra, percentile_intra), 2)

    if percentile_inter:
        coexpression_inter = coexpression_full * np.logical_not(mask)
        coexpression_inter[coexpression_inter == 0] = np.nan
        threshold_inter = np.round(np.nanpercentile(coexpression_inter, percentile_inter), 2)

    if percentile_intra and percentile_inter:
        return threshold_intra, threshold_inter
    elif percentile_intra:
        return threshold_intra
    else:
        return threshold_inter 


def chromatin_threshold(hic, mask=None, percentile_intra=None, percentile_inter=None):
    if not percentile_intra and not percentile_inter:
        raise ValueError(
            'Either one parameter between percentile_intra and percentile_inter must be different from zero.')

    if percentile_inter:
        if percentile_intra:
            hic_intra = hic * mask
            hic_intra[hic_intra == 0] = np.nan
            threshold_intra = np.round(np.nanpercentile(hic_intra, percentile_intra), 2)

        hic_inter = hic * np.logical_not(mask)
        hic_inter[hic_inter == 0] = np.nan
        threshold_inter = np.round(np.nanpercentile(hic_inter, percentile_inter), 2)
    else:
        threshold_intra = np.round(np.nanpercentile(hic[hic > 0], percentile_intra), 2)

    if percentile_intra and percentile_inter:
        return threshold_intra, threshold_inter
    elif percentile_intra:
        return threshold_intra
    else:
        return threshold_inter

def build_full_hic(path, dataset, type,window):
    if not os.path.exists('{}/{}/hic/{}_all_{}.npy'.format(path, dataset, type, window)):
        rows = []
        for i in range(1, 23):
            row = []
            for j in range(1, 23):

                if i <= j:
                    hic = np.load(
                    '{}/{}/hic/{}_{}_{}_{}.npy'.format(path, dataset, type, i, j, window))
                    row.append(hic)
                else:
                    hic = np.load(
                        '{}/{}/hic/{}_{}_{}_{}.npy'.format(path, dataset, type, j, i, window)).T
                    hic = np.empty(hic.shape)
                    hic[:] = np.nan
                    row.append(hic)
            rows.append(np.hstack(row))
        hic_full = np.vstack(rows)
        np.save('{}/{}/hic/{}_all_{}.npy'.format(path, dataset, type, window), hic_full)
        print('Full genome-wide Hi-C saved in {}/{}/hic/{}_all_{}.npy'.format(path, dataset, type, window))
    else:
        hic_full = np.load('{}/{}/hic/{}_all_{}.npy'.format(path, dataset, type, window))
    return hic_full