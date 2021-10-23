import networkx as nx
import torch
import numpy as np
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


""" 
The function in this file is modified based on the https://github.com/LarsHoldijk/RE-ParameterizedExplainerForGraphNeuralNetworks
"""

def plot(graph, edge_weigths, labels, thres_min, thres_snip, dataset, expl_log_dir, idx=None, idx_adj=None, idx_label=None, name=None, args=None, show=False):
    """
    Function that can plot an explanation (sub)graph and store the image.

    :param graph: graph provided by explainer
    :param edge_weigths: Mask of edge weights provided by explainer
    :param labels: Label of each node required for coloring of nodes
    :param idx: Node index of a interesting node pair
    :param idx_adj: contact between the node pair
    :param idx_label: co-expression between the node pair
    :param thresh_min: (total number of edges) selected threshold
    :param thres_snip: number of top edges
    :param dataset: the name of the dataset
    :param expl_log_dir: ouput path
    :param args: Object containing arguments from configuration
    :param show: flag to show plot made
    """
    if idx is not None:
        id0, id1 = int(idx[0]), int(idx[1])

    # Set thresholds
    sorted_edge_weigths = np.sort(edge_weigths)

    thres_index = max(int(edge_weigths.shape[0]-thres_snip),0)

    thres = sorted_edge_weigths[thres_index]

    if thres < thres_min:
        thres = thres_min
    filter_thres = thres_min

    # Init edges
    filter_nodes = set()
    filter_edges = []
    pos_edges = []
    edge_label = []
    weight_edges = []
    # Select all edges and nodes to plot
    for i in range(edge_weigths.shape[0]):
        # Select important edges
        if edge_weigths[i] >= thres and not graph[0][i] == graph[1][i]:
            pos_edges.append((graph[0][i].item(),graph[1][i].item()))
            edge_label.append(int(labels[i]))
            weight_edges.append(edge_weigths[i].item())
        # Select all edges to plot
        if edge_weigths[i] > filter_thres and not graph[0][i] == graph[1][i]:
            filter_edges.append((graph[0][i].item(),graph[1][i].item(),edge_weigths[i]))
            filter_nodes.add(graph[0][i].item())
            filter_nodes.add(graph[1][i].item())
            
    pos_nodes = np.unique(pos_edges)
    num_nodes = len(pos_nodes)
    # Initialize graph object
    G = nx.Graph()

    if not (thres_min == -1):
        # Deal with plotting of node datasets
        G.add_weighted_edges_from(filter_edges)
        G.remove_nodes_from(list(nx.isolates(G)))
        if idx is not None:
            if id0 not in G.nodes():
                G.add_node(id0)
            if id1 not in G.nodes():
                G.add_node(id1)
        pos = nx.nx_pydot.graphviz_layout(G)# for large undirected graphs , prog='fdp'

        if idx is not None:
            for cc in nx.connected_components(G):
                if id0 in cc and id1 in cc:
                    G = G.subgraph(cc).copy()
                    break
            G.remove_nodes_from(list(nx.isolates(G)))    
        
        # ipdb.set_trace()
        pos_edges = [(u, v) for (u, v) in pos_edges if u in G.nodes() and v in G.nodes()]
        edge_colors = ['#2c7bb6', '#d6604d']
        color = cm.get_cmap('PuOr', len(G.nodes()))
        cmap = color(np.linspace(0.9, 0.1, 2*len(name)))
        cmap = cmap[::2]
        node_colors = [cmap[n] for n in G.nodes()]

        # node_colors = [i for i in range(len(filter_nodes))]
        # edge coloring
        label2edges = []
        weight2edges = []
        max_edge_label = np.max(edge_label) + 1
        if -1 in edge_label:
            max_edge_label += 1
            edge_colors.append('#252525')
        nmb_edges = len(pos_edges)

        # Create empty lists of possible labels
        for i in range(max_edge_label):
            label2edges.append([])
            weight2edges.append([])
        
        for i in range(nmb_edges):
            # (u, v) = filt_eg_lt[i]
            label2edges[edge_label[i]].append(pos_edges[i])
            weight2edges[edge_label[i]].append(weight_edges[i]*5)
        
        # Draw an edge
        if idx is not None:
            fig, ax = plt.subplots(1,1, figsize=(26, 28))
            nsz = 1400
            fsz = 44
            lfsz = 40
        else:
            fig, ax = plt.subplots(1, 1, figsize=(36, 30)) #(22,18c) 
            lfsz = 40
        edges = G.edges()
        weights = [G[u][v]['weight']*5 for u,v in edges]
        

        for i in range(len(label2edges)):
            weight_list = weight2edges[i]
            edge_list = label2edges[i]
            if len(edge_list) == 0:
                continue
            node_list = np.unique(edge_list)
            edge_color = edge_colors[i % len(edge_colors)]
            alpha = 0.8
            if idx is not None:
                alpha = 0.6 if edge_color == '#2c7bb6' else 0.9
            
            # Draw all nodes of a ''certain'' color
            nx.draw_networkx_edges(G,
                                    pos,
                                    width=weight_list,
                                    edgelist=edge_list,
                                    edge_color=edge_color,
                                    alpha=alpha)
            nx.draw_networkx_nodes(G, pos,
                                    nodelist=list(node_list),
                                    node_color=cmap[node_list],
                                    node_size=1000, 
                                    edgecolors='black',linewidths=1.5)
            labels = {}
            if idx is not None:
                if len(label2edges) <= 1:
                    return None
                pos_posi_nodes = np.unique(label2edges[1])
                
                for i in range(len(weight_list)):
                    i0, i1 = edge_list[i]
                    if i0 in pos_posi_nodes and i1 in pos_posi_nodes:
                        if i0 != id0:
                            labels[i0] = name[i0]
                        if i1 != id1:
                            labels[i1] = name[i1]
            else:
                th = np.percentile(np.hstack(weight2edges), 87)
                for i in range(len(weight_list)):
                    w = weight_list[i]
                    i0, i1 = edge_list[i]
                    if w >= th and i0 in pos_nodes and i1 in pos_nodes:
                        labels[i0] = name[i0]
                        labels[i1] = name[i1]

            nx.draw_networkx_labels(G, pos, labels, font_family='Arial', font_size=lfsz)
        # Draw the base nodes
        
        if idx is not None:
            nx.draw_networkx_nodes(G, pos,
                                nodelist=[id0],
                                node_color=cmap[id0].reshape(1,-1),
                                node_size=nsz, 
                                edgecolors='black',linewidths=1)
            nx.draw_networkx_labels(G,pos, {id0: name[id0]},font_size=fsz,font_weight='bold')
        
            nx.draw_networkx_nodes(G, pos,
                                    nodelist=[id1],
                                    node_color=cmap[id1].reshape(1,-1),
                                    node_size=nsz, 
                                    edgecolors='black',linewidths=1)
            nx.draw_networkx_labels(G,pos, {id1: name[id1]},font_size=fsz,font_weight='bold')

    # Deal with plotting of graph datasets
    else:
        # Format edges
        edges = [(pair[0], pair[1]) for pair in gt[0][idx].T]
        # Obtain all unique nodes
        nodes = np.unique(gt[0][idx])
        # Add all unique nodes and all edges
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        # Let the graph generate all positions
        pos = nx.kamada_kawai_layout(G)

        pos_edges = [(u, v) for (u, v) in pos_edges if u in G.nodes() and v in G.nodes()]

        nx.draw_networkx_nodes(G,
                               pos,
                               nodelist=nodes,
                               node_color='red',
                               node_size=500)

    ax= plt.gca()
    if idx is not None:
        ax.set_title('Local attention subgraph for ({}, {}, {}, {}) (chr{})'.format(name[id0], name[id1], int(idx_adj), int(idx_label), args.chr_src), fontdict={'family':'Arial', 'fontsize': 14})
    else:
        ax.set_title('Global attention subgraph for chr{}'.format(args.chr_src), fontdict={'family':'Arial', 'fontsize': 14})
    plt.axis('off')
    # ax.collections[0].set_edgecolor("black") # set node border
    
    bounds0 = np.linspace(1, len(name), len(name)) #len(name)+1
    if idx is not None:
        tck_num = 10
    else:
        tck_num = 20
    bounds1 = np.linspace(1, len(name), tck_num) #int(len(name)/50)
    cmap = mpl.cm.PuOr_r
    norm = mpl.colors.BoundaryNorm(bounds0, len(bounds0))

    ax2 = fig.add_axes([0.92, 0.15, 0.02, 0.7]) #[left, bottom, width, height]
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=None, spacing='proportional', ticks=bounds1, boundaries=bounds0, format='%1i')
    cb.ax.tick_params(labelsize=26)
    ax2.set_ylabel('Color range of gene nodes', size=26, fontdict={'family': 'Arial'})
    # ipdb.set_trace()
    # cax = plt.colorbar(nodes, cmap=cmap, ticks=np.arange(np.min(nodes), np.max(nodes)+1, 10))

    save_path = '{}/subgraphs'.format(expl_log_dir)
    # Generate folders if they do not exist
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Save figure 
    if idx is not None:
        plt.savefig('{}/Local_attention_chr_{}_{}_({}_{}).pdf'.format(save_path, args.chr_src, args.chr_tgt, id0, id1), dpi=600)
    else:
        plt.savefig('{}/Global_attention_chr_{}_{}.pdf'.format(save_path, args.chr_src, args.chr_tgt), dpi=600)
    
    plt.clf()
    if show:
        plt.show()
    
    edge_11 = np.stack(label2edges[1])
    node_11 = np.unique(edge_11).tolist()
    if idx is not None:
        node_11.extend([id0, id1])
    return node_11 