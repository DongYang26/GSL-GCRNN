import torch
import torch.nn as nn
from models.modules.GraphConvCell import GraphConvCell

# torch.manual_seed(719)


class GcRNNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(GcRNNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        
        self.GraphConvCell_1 = GraphConvCell(self._hidden_dim, self._hidden_dim*2, bias=1.0)
        self.GraphConvCell_2 = GraphConvCell(self._hidden_dim, self._hidden_dim)

        self.projection = nn.Linear(self._hidden_dim, 1)

    def forward(self, inputs, hidden_state, view):
        # [r, u] = sigmoid(A[x, h]W + b)
        concatenation = torch.sigmoid(self.GraphConvCell_1(inputs, hidden_state,view))  
        r, u = torch.chunk(concatenation, chunks=2, dim=1)  
        # c = tanh(A[x, (r * h)W + b])
        c = torch.tanh(self.GraphConvCell_2(inputs, r * hidden_state,view))  
        # h = u * h + (1 - u) * c
        new_hidden_state = u * hidden_state + (1.0 - u) * c  
        return new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}



