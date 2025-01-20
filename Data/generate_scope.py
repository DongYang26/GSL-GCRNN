import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd


def get_khop_indices(k, view):
    view = (view.A > 0).astype("int32")
    view_ = view
    for i in range(1, k):
        view_ = (np.matmul(view_, view.T)>0).astype("int32")
    view_ = torch.tensor(view_).to_sparse()
    return view_.indices()
    
def topk(k, adj):
    pos = np.zeros(adj.shape)
    for i in range(len(adj)):
      one = adj[i].nonzero()[0]
      if len(one)>k:
        oo = np.argsort(-adj[i, one])
        sele = one[oo[:k]]
        pos[i, sele] = adj[i, sele]
      else:
        pos[i, one] = adj[i, one]
    return pos


def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    adj = sp.coo_matrix(adj)
    return adj

#####################
## get k-hop scope ##
## take citeseer   ##
#####################
data_name = "./trainData/bcn_L"
adj = sp.load_npz(data_name+"/v2_adj.npz")
indice = get_khop_indices(1, adj)
torch.save(indice, data_name+"/v2_2.pt")

#####################
## get top-k scope ##
## take citeseer   ##
#####################
adj = sp.load_npz(data_name+"/v3_diff.npz")
kn = topk(5, adj.toarray())
kn = sp.coo_matrix(kn)
indice = get_khop_indices(1, kn)
torch.save(indice, data_name+"/v3_5.pt")
