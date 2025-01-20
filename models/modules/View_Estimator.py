import torch
import torch.nn as nn
from utils.view_normalize import normalize


class GCN_one(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True, activation=None):
        super(GCN_one, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fc = nn.Linear(in_ft, out_ft, bias=False).to(self.device)
        self.activation = activation
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft)).to(self.device)
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, feat, adj):
        feat = feat.to(self.device)
        adj = adj.to_dense().to(self.device)
        feat_fc = self.fc(feat)
        out = torch.spmm(adj, feat_fc).to(self.device)
        if self.bias is not None:
            out += self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out


class GenView(nn.Module):
    def __init__(self, num_feature, hid, com_lambda, dropout):
        super(GenView, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gen_gcn = GCN_one(num_feature, hid, activation=nn.ReLU()).to(self.device)
        self.gen_mlp = nn.Linear(2 * hid, 1).to(self.device)
        nn.init.xavier_normal_(self.gen_mlp.weight, gain=1.414)
        self.relu = nn.ReLU().to(self.device)
        self.softmax = nn.Softmax(dim=1).to(self.device)

        self.com_lambda = com_lambda
        self.dropout = nn.Dropout(dropout).to(self.device)

    def forward(self, v_ori, feat, v_indices, num_node):
        emb = self.gen_gcn(feat, v_ori)
        f1 = emb[v_indices[0]]
        f2 = emb[v_indices[1]]
        ff = torch.cat([f1, f2], dim=-1)
        temp = self.gen_mlp(self.dropout(ff)).reshape(-1)
        z_matrix = torch.sparse.FloatTensor(v_indices.to(self.device), temp.to(self.device), (num_node, num_node))
        pi = torch.sparse.softmax(z_matrix, dim=1).to(self.device)
        v_ori = v_ori.to(self.device)
        gen_v = v_ori + self.com_lambda * pi
        return gen_v


class View_Estimator(nn.Module):
    def __init__(self, num_feature, gen_hid, com_lambda_v1, com_lambda_v2, dropout):
        super(View_Estimator, self).__init__()
        self.v1_gen = GenView(num_feature, gen_hid, com_lambda_v1, dropout)
        self.v2_gen = GenView(num_feature, gen_hid, com_lambda_v2, dropout)

    def forward(self, data, num_node, inputs):
        feat = inputs.transpose(0, 1).to(self.v1_gen.device)
        new_v1 = normalize(self.v1_gen(data.view_1, feat, data.indices_v1, num_node))
        new_v2 = normalize(self.v2_gen(data.view_2, feat, data.indices_v2, num_node))
        return new_v1, new_v2
