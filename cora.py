import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
# from gcn import GCNNet as Net
# from gcn_pyg import GCNNet as Net
# from gat_pyg import GATNet as Net
from sage_pyg import SAGENet as Net

EPOCH = 100


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test(model, data):
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def main():
    dataset = 'Cora'
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', dataset)
    dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = Net(dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, EPOCH):
        train(model, data, optimizer)
        accs = test(model, data)
        log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, *accs))


if __name__ == '__main__':
    main()