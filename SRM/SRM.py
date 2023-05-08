import torch
import torch.nn as nn


class SRMLayer(nn.Module):
    def __init__(self):
        super(SRMLayer, self).__init__()
        self.q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        filter1 = torch.tensor(filter1, dtype=torch.float) / self.q[0]
        filter2 = torch.tensor(filter2, dtype=torch.float) / self.q[1]
        filter3 = torch.tensor(filter3, dtype=torch.float) / self.q[2]
        filters = torch.stack([filter1, filter2, filter3], dim=2)  # shape=(5,5,3)

        initializer_srm = nn.init.constant_(nn.Parameter(filters), filters)
        self.conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)  # same填充方式填充kernel_size/2（向下取整）；valid不填充
        self.conv.weight = initializer_srm

    def forward(self, x):
        return self.conv(x)
