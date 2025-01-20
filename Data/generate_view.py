import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power, inv
import pandas as pd


def knn(feat, num_node, k, data_name, view_name):
    adj = np.zeros((num_node, num_node), dtype=np.int64)
    dist = cos(feat)
    col = np.argpartition(dist, -(k + 1), axis=1)[:, -(k + 1):].flatten()
    adj[np.arange(num_node).repeat(k + 1), col] = 1
    adj = sp.coo_matrix(adj)
    sp.save_npz("./"+data_name+"/"+view_name+"_knn_"+str(k)+".npz", adj)


def adja(adj, data_name, view_name):
    adj = sp.coo_matrix(adj)
    sp.save_npz("./"+data_name+"/"+view_name+"_adj.npz", adj)


def diff(adj, alpha, data_name, view_name):   
    d = np.diag(np.sum(adj, 1))                                    
    dinv = fractional_matrix_power(d, -0.5)                       
    at = np.matmul(np.matmul(dinv, adj), dinv)                      
    adj = alpha * inv((np.eye(adj.shape[0]) - (1 - alpha) * at))   
    adj = sp.coo_matrix(adj)
    sp.save_npz("./"+data_name+"/"+view_name+"_diff.npz", adj)


def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    feat = sp.coo_matrix(feat)
    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    adj = sp.coo_matrix(adj)
    return adj


data_name = "trainData/bcn_L"  # first
view_name = "v2"  # v1 or v2 or v3
view_type = "adj"  # knn or adj or diff
k = 2
alpha = 0.5


# # 
# with open("./" + data_name + "/16991_adj_bcn_L__0.9_spearman.csv", 'r') as f:
#     lines = f.readlines()
# # 
# lines = lines[1:]
# # 
# with open("./" + data_name + "/16991_adj_bcn_L__0.9_spearman.csv", 'w') as f:
#     f.writelines(lines)


adj = load_adjacency_matrix("./"+data_name+"/101_adj_bcn_L__0.5_spearman.csv")
print("adj shape：", adj.shape)
num_node = adj.shape[0]
feat = load_features("./"+data_name+"/bcn-L_202106010800-202106080800_1mins_2MHz.csv")
feat = feat.transpose()
a = adj.A
if a[0, 0] == 0:
    a += np.eye(num_node)
    print("self-loop!")
adj = a
if view_type == "knn":  # set k
    knn(feat, num_node, k, data_name, view_name)
elif view_type == "adj":
    adja(adj, data_name, view_name)
elif view_type == "diff":  # set alpha: 0~1
    diff(adj, alpha, data_name, view_name)
