import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GNN.GNNModelBase import GNNModelBase
from dgl.nn.pytorch import GATConv
from dgl import DGLGraph



class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for _ in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu, 
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)



class HAN(GNNModelBase):
    def __init__(self, meta_paths, n_heads, residual, **kwargs):
        super().__init__(**kwargs)

        self.layers = nn.ModuleList()
        #self.layers.append(HANLayer(num_meta_paths, self.hidden_dim, self.hidden_size/n_heads, n_heads, self.p_dropout))
        for l in range(self.n_layers):
            self.layers.append(HANLayer(meta_paths, self.hidden_dim, 
                                        self.hidden_dim // n_heads, n_heads, self.p_dropout))
        #self.predict = nn.Linear(hidden_size * n_heads, out_size)

    #def forward(self, g, h):
    #    for gnn in self.layers:
    #        h = gnn(g, h)

    #    return self.predict(h)
    
    def gnn_forward(self, g:DGLGraph, fz_embedding, main_node_ids):
        feats = g.ndata['h']
        for layer in self.layers:
            feats = layer(g, feats)
            #feats = feats.reshape(feats.shape[0], -1)
        
        if self.use_readout:
            readout = self.readout(g, feats)
        else:
            readout = feats[main_node_ids,:]
        if self.cat_fz_embedding:
            readout = torch.cat((readout, fz_embedding), 1)
        out = self.fcout(readout)
        return out

# meta_paths=[['pa', 'ap'], ['pf', 'fp']],