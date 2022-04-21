import torch
import torch_geometric as ptgeom

def subgraph(graph, pairs):
    """
    Given the index of a pair of genes this method returns the union of two 1-hop subgraphs. 
    :param index: index of the node/graph that we wish to explain
    :return: union  graph
    """       

    # we only consider the union of two genes' subgraph for explaining
    n0, n1 = pairs
    n0 = int(n0)
    n1 = int(n1)
    pairs = pairs.reshape(1,-1)
    graph = torch.LongTensor(graph)
    graph0 = ptgeom.utils.k_hop_subgraph(n0, 1, graph)[1]
    graph1 = ptgeom.utils.k_hop_subgraph(n1, 1, graph)[1]
    local_graph = torch.unique(torch.hstack((graph0,graph1)), sorted=False, dim=1)
    # only take the union of two nodes' ego-subgraph without the edges between neighbors.
    idx=torch.where((local_graph==n0) | (local_graph==n1))[1].unique()
    local_graph = local_graph[:,idx]
        
    return local_graph.numpy()