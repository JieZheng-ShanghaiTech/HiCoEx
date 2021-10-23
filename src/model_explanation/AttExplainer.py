import torch
import torch_geometric as ptgeom
import numpy as np


class AttExplainer(object):
    def __init__(self, model_to_explain, graph, features):
        self.model_to_explain = model_to_explain
        self.graph = graph
        self.features = features

    def explain(self, pairs, th, task):
        """
        Given the index of a pair of genes this method returns its explanation. 
        :param index: index of the node/graph that we wish to explain
        :return: explanaiton graph and edge weights
        """
        # ipdb.set_trace()
        self.model_to_explain.eval()
        with torch.no_grad():
            feats = self.features
            graph = self.graph          

            if task == 'local':
                # we only consider the union of two genes' subgraph for explaining
                n0, n1 = pairs
                n0 = int(n0)
                n1 = int(n1)
                pairs = pairs.reshape(1,-1)
                graph0 = ptgeom.utils.k_hop_subgraph(n0, 1, self.graph)[1]
                graph1 = ptgeom.utils.k_hop_subgraph(n1, 1, self.graph)[1]
                graph = torch.unique(torch.hstack((graph0,graph1)), dim=1)
                _, alpha = self.model_to_explain(feats, graph)
            
            att_embeds, alpha = self.model_to_explain(feats, graph)
            att_pred = self.model_to_explain.edge_predictor(att_embeds, pairs)

            th_a = np.percentile(alpha.detach().cpu(), th)
            alpha_idx = torch.where(alpha > th_a)[0]
            att_graph = graph[:,alpha_idx]

            original_embeds, alpha = self.model_to_explain(feats, self.graph)
            original_pred = self.model_to_explain.edge_predictor(original_embeds, pairs)
        
        return alpha, graph, att_graph, original_pred, att_pred