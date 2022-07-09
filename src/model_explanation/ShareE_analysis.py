import numpy as np
import pandas as pd
import ipdb

def shareE_analysis_atlas(chrom, enhancers, signifi_shared, signifi_genes, pair_list, local_dict, mcf7_rna, mcf7_coexp, random=False, skip=False):
    mcf7_E_all = enhancers[enhancers['chrom'] == 'chr'+chrom]
    
    mcf7_coexp_ = np.nan_to_num(mcf7_coexp)
    mcf7_coexp1 = mcf7_coexp_ + mcf7_coexp_.T
    
    shared_dict_all = {}  # all common neighbors, attention, and co-exp labels 
    share_dict = {}       # common neighbors in shared_dict_all
    share_dict_E = {}     # enhancers shared by neighbors and genes
    share_dict_gene = {}  # the set of all gene-neighbors that share same enhancer in share_dict_E with a gene or both two genes
    share_dict_itset = {}  # neighbor set that shared enhancers with both of two genes
    share_itset_nei = {}
    x1 = []
    for i in range(len(signifi_shared)):
        (u,v) = pair_list[signifi_shared[i][0]]
        mask = np.isin(local_dict[(u,v)], signifi_shared[i][1])
        shared_dict = local_dict[(u,v)][mask[:,1]]

        u_name = mcf7_rna.loc[u, 'Gene name']
        v_name = mcf7_rna.loc[v, 'Gene name']
        
        # skip the gene pairs which already contact with same enhancers
        if skip:
            mcf7_E_uv = mcf7_E_all[mcf7_E_all['gene'].isin([u_name, v_name])]
            mcf7_E_uv.loc[:,'enhancer'] = mcf7_E_uv['enhancer'].astype(str)
            if len(mcf7_E_uv) < 2 or (mcf7_E_uv['enhancer'].isna()).sum() > 0:
                continue

            E_uv = [e.split(';') for e in mcf7_E_uv['enhancer']]
            E_count_uv = pd.value_counts(np.hstack(E_uv))
            if (E_count_uv==2).sum() > 0:
                continue
            
        shared_dict_all[(u_name, v_name)] = shared_dict
        r,c = shared_dict.T
        j1 = c
        uv_coexp = mcf7_coexp1[r,c]
        uv_11_id = np.where(uv_coexp==1)[0]
        all_com_nei = np.setdiff1d(np.unique(j1),[u,v])
        
                    
        share_set = signifi_genes[i].tolist()        
        mcf7_E0 = mcf7_E_all[mcf7_E_all['gene'].isin(share_set)]
        mcf7_E1 = mcf7_E0.loc[~mcf7_E0['enhancer'].isna()]
        mcf7_E1.loc[:,'enhancer'] = mcf7_E1['enhancer'].astype(str)

        E = [e.split(';') for e in mcf7_E1['enhancer']]
        if len(E) == 0:
            continue
        E_count = pd.value_counts(np.hstack(E))
        share_E = E_count[E_count>=2]      

        share_E1_u, share_E1_v = [],[]
        share_E1_geneu,share_E1_genev = [],[]
        for e in share_E.index.values:
            E_gene = mcf7_E1.loc[mcf7_E1['enhancer'].str.contains(e),'gene']
            if u_name in E_gene.tolist():
                share_E1_u.append(e)
                share_E1_geneu.append(E_gene.tolist())
            if v_name in E_gene.tolist():
                share_E1_v.append(e)
                share_E1_genev.append(E_gene.tolist())

#             if len(share_E1_geneu) > 0 and len(share_E1_genev) > 0:
        if len(share_E1_geneu) == 0 or len(share_E1_genev) == 0:
            x1.append(0)
        else:
            share_E1_setu = np.unique(np.hstack(share_E1_geneu))
            share_E1_setv = np.unique(np.hstack(share_E1_genev))
            share_E1_setu = share_E1_setu[share_E1_setu != u_name]
            share_E1_setv = share_E1_setv[share_E1_setv != v_name]
            share_E1_uv = np.intersect1d(share_E1_setu, share_E1_setv)

            tmp = share_E1_u + share_E1_v
            tmp1 = share_E1_geneu + share_E1_genev

            share_set.remove(u_name)
            share_set.remove(v_name)    
            share_dict[(u_name,v_name)] = share_set
            share_dict_E[(u_name,v_name)] = tmp
            share_dict_gene[(u_name,v_name)] = tmp1
            share_dict_itset[(u_name,v_name)] = share_E1_uv
            share_itset_nei[(u_name,v_name)] = all_com_nei
            # -------------statistical analysis--------------           
            j2 = mcf7_rna[mcf7_rna['Gene name'].isin(share_E1_uv)].index.values.tolist()

            share_E_coexp = len(np.unique(j2))
            x1.append(share_E_coexp/len(all_com_nei))                  


    share_all = (shared_dict_all, share_dict, share_dict_E, share_dict_gene, share_dict_itset, share_itset_nei)
    return share_all, x1