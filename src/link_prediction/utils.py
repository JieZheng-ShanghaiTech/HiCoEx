import os
import numpy as np
import torch

def intra_mask(shapes, nans=False, values=np.ones):
    mask = np.zeros(np.sum(shapes, axis=0))
    if nans:
        mask[:] = np.nan

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        mask[r:r + rr, c:c + cc] = values((rr, cc))
        r += rr
        c += cc

    return mask

def block_matrix(arrs):
    shapes = np.array([a.shape for a in arrs])
    full = np.zeros(np.sum(shapes, axis=0))
    full[:] = np.nan
    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        full[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc

    return full


def set_gpu(active=False, gpu_id=0):
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(gpu_id)
    if torch.cuda.is_available() and active:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")            
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device



def set_n_threads(n_threads):
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
