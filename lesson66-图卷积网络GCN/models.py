import  torch.nn as nn
import  torch.nn.functional as F
from    layers import GraphConvolution


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        """

        :param x: [2708, 1433]
        :param adj: [2708, 2708]
        :return:
        """
        # print('x:', x.shape, 'adj:', adj.shape)
        # => [2708, 16]
        x = F.relu(self.gc1(x, adj))
        # print('gcn1:', x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        # => [2708, 7]
        x = self.gc2(x, adj)
        # print('gcn2:', x.shape)
        return F.log_softmax(x, dim=1)
