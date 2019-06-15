from sys import exit
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean, scatter_add


class MetaLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, e_h, n_h):
        super(MetaLayer, self).__init__()
        self.edge_mlp = Seq(torch.nn.Linear(2 * n_in, e_h),
                            ReLU()
                            )
        self.node_mlp_1 = Seq(torch.nn.Linear(e_h + n_in, n_h),
                              ReLU()
                              )
        self.node_mlp_2 = Seq(torch.nn.Linear(n_h, n_out),
                              ReLU()
                              )

    def _n2e_e2e(self, src, dest):
        out = torch.cat([src, dest], 1)
        out = self.edge_mlp(out)
        return out

    def _e2n_n2n(self, x, edge_index, edge_attr):
        src, dest = edge_index
        out = torch.cat([x[dest], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_add(out, dest, dim=0, dim_size=x.size(0))
        out = self.node_mlp_2(out)
        return out

    def forward(self, x, edge_index):
        edge_src, edge_dest = edge_index
        edge_attr = self._n2e_e2e(x[edge_src], x[edge_dest])
        x = self._e2n_n2n(x, edge_index, edge_attr)
        return x


class MetaNet_test(torch.nn.Module):
    def __init__(self, dataset):
        super(MetaNet, self).__init__()
        self.meta1 = MetaLayer(dataset.num_features, 128, 128, 128)
        self.meta2 = MetaLayer(128, dataset.num_classes, 128, 128)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.meta1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.meta2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


class MetaNet(torch.nn.Module):
    def __init__(self, dataset):
        super(MetaNet, self).__init__()
        self.meta1 = MetaLayer(dataset.num_features, dataset.num_classes, 128, 128)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.meta1(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = F.log_softmax(x, dim=1)
        return x