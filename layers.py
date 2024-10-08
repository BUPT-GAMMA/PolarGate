# coding=utf-8
from typing import Union
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.typing import PairTensor, Adj
from torch_sparse import matmul
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from torch_sparse import coalesce


_norm_layer_factory = {
    'batchnorm': nn.BatchNorm1d,
}

_act_layer_factory = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'sigmoid': nn.Sigmoid,
}


def create_spectral_features(
        pos_edge_index: torch.LongTensor,
        neg_edge_index: torch.LongTensor,
        node_num: int,
        dim: int
) -> torch.FloatTensor:
    edge_index = torch.cat(
        [pos_edge_index, neg_edge_index], dim=1)
    N = node_num
    edge_index = edge_index.to(torch.device('cpu'))

    pos_val = torch.full(
        (pos_edge_index.size(1),), 2, dtype=torch.float)
    neg_val = torch.full(
        (neg_edge_index.size(1),), 0, dtype=torch.float)
    val = torch.cat([pos_val, neg_val], dim=0)

    row, col = edge_index
    edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
    val = torch.cat([val, val], dim=0)

    edge_index, val = coalesce(edge_index, val, N, N)
    val = val - 1

    edge_index = edge_index.detach().numpy()
    val = val.detach().numpy()
    A = sp.coo_matrix((val, edge_index), shape=(N, N))
    svd = TruncatedSVD(n_components=dim, n_iter=128)
    svd.fit(A)
    x = svd.components_.T
    return torch.from_numpy(x).to(torch.float)


class MLP(nn.Module):
    def __init__(self, dim_in=256, dim_hidden=32, dim_pred=1, num_layer=3, norm_layer=None, act_layer=None, p_drop=0.5,
                 sigmoid=False, tanh=False):
        super(MLP, self).__init__()
        '''
        The basic structure is refered from 
        '''
        assert num_layer >= 2, 'The number of layers shoud be larger or equal to 2.'
        if norm_layer in _norm_layer_factory.keys():
            self.norm_layer = _norm_layer_factory[norm_layer]
        if act_layer in _act_layer_factory.keys():
            self.act_layer = _act_layer_factory[act_layer]
        if p_drop > 0:
            self.dropout = nn.Dropout

        fc = []
        # 1st layer
        fc.append(nn.Linear(dim_in, dim_hidden))
        if norm_layer:
            fc.append(self.norm_layer(dim_hidden))
        if act_layer:
            fc.append(self.act_layer(inplace=True))
        if p_drop > 0:
            fc.append(self.dropout(p_drop))
        for _ in range(num_layer - 2):
            fc.append(nn.Linear(dim_hidden, dim_hidden))
            if norm_layer:
                fc.append(self.norm_layer(dim_hidden))
            if act_layer:
                fc.append(self.act_layer(inplace=True))
            if p_drop > 0:
                fc.append(self.dropout(p_drop))
        # last layer
        fc.append(nn.Linear(dim_hidden, dim_pred))
        # sigmoid
        if sigmoid:
            fc.append(nn.Sigmoid())
        if tanh:
            fc.append(nn.Tanh())
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        out = self.fc(x)
        return out


class PolarGateConv(MessagePassing):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        first_aggr: bool,
        bias: bool = True,
        norm_emb: bool = False,
        **kwargs
    ):

        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.first_aggr = first_aggr
        self.norm_emb = norm_emb

        if first_aggr:
            self.lin_b = Linear(2 * in_dim, out_dim, bias)
            self.lin_u = Linear(2 * in_dim, out_dim, bias)
        else:
            self.lin_b = Linear(3 * in_dim, out_dim, bias)
            self.lin_u = Linear(3 * in_dim, out_dim, bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_b.reset_parameters()
        self.lin_u.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: Adj,
                neg_edge_index: Adj) -> Tensor:

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        if self.first_aggr:
            out_b = self.propagate(pos_edge_index, x=x)
            out_b = self.lin_b(torch.cat([out_b, x[1]], dim=-1))

            out_u = self.propagate(neg_edge_index, x=x)
            out_u = self.lin_u(torch.cat([out_u, x[1]], dim=-1))
            out = torch.cat([out_b, out_u], dim=-1)
        else:
            F_in = self.in_dim
            out_b1 = self.propagate(pos_edge_index, x=(
                x[0][..., :F_in], x[1][..., :F_in]))
            out_b2 = self.propagate(neg_edge_index, x=(
                x[0][..., F_in:], x[1][..., F_in:]))
            out_b = torch.cat([out_b1, out_b2, x[1][..., :F_in]], dim=-1)
            out_b = self.lin_b(out_b)

            out_u1 = self.propagate(pos_edge_index, x=(
                x[0][..., F_in:], x[1][..., F_in:]))
            out_u2 = self.propagate(neg_edge_index, x=(
                x[0][..., :F_in], x[1][..., :F_in]))
            out_u = torch.cat([out_u1, out_u2, x[1][..., F_in:]], dim=-1)
            out_u = self.lin_u(out_u)

            out = torch.cat([out_b, out_u], dim=-1)
        if self.norm_emb:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: PairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_dim}, '
                f'{self.out_dim}, first_aggr={self.first_aggr})')


class restPolarGateConv(MessagePassing):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        first_aggr: bool = False,
        bias: bool = True,
        norm_emb: bool = False,
        **kwargs
    ):

        kwargs.setdefault('aggr', 'min')
        # kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.first_aggr = first_aggr
        self.norm_emb = norm_emb

        self.lin_b = Linear(3 * in_dim, out_dim, bias)
        self.lin_u = Linear(3 * in_dim, out_dim, bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_b.reset_parameters()
        self.lin_u.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: Adj,
                neg_edge_index: Adj) -> Tensor:

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        F_in = self.in_dim

        # update positive embeddings
        out_b1 = self.propagate(pos_edge_index, x=(
            x[0][..., :F_in], x[1][..., :F_in]))
        out_b2 = self.propagate(neg_edge_index, x=(
            x[0][..., F_in:], x[1][..., F_in:]))
        out_b = torch.cat([out_b1, out_b2 * -1, x[1][..., :F_in]], dim=-1)
        out_b = self.lin_b(out_b)

        # update negative embeddings
        out_u1 = self.propagate(pos_edge_index, x=(
            x[0][..., F_in:], x[1][..., F_in:]))
        out_u2 = self.propagate(neg_edge_index, x=(
            x[0][..., :F_in], x[1][..., :F_in]))
        out_u = torch.cat([out_u1, out_u2 * -1, x[1][..., F_in:]], dim=-1)
        out_u = self.lin_u(out_u)

        out = torch.cat([out_b, out_u], dim=-1)
        if self.norm_emb:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: PairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_dim}, '
                f'{self.out_dim}, first_aggr={self.first_aggr})')


