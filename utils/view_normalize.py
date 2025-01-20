import torch


def normalize(adj, data=False):
    if data:
        adj_ = (adj + adj.t())
        normalized_adj = adj_
    else:
        adj_ = (adj + adj.t())
        normalized_adj = _normalize(adj_ + torch.eye(adj_.shape[0]).to(adj.device).to_sparse())
    return normalized_adj


def _normalize(mx):
    mx = mx.to_dense()
    rowsum = mx.sum(1) + 1e-6  # avoid NaN
    r_inv = rowsum.pow(-1 / 2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx.to_sparse()
