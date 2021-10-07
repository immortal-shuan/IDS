import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepNet(nn.Module):
    def __init__(self, input_size, num_class, dropout):
        super(DeepNet, self).__init__()

        self.res_layer = nn.Sequential(nn.Linear(input_size, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, input_size),
                                       nn.ReLU())

        self.norm = nn.Sequential(nn.Linear(2*input_size, input_size),
                                  nn.BatchNorm1d(input_size))

        self.dropout = nn.Dropout(dropout)

        self.classi_layer = nn.Sequential(nn.Linear(input_size, 1024),
                                          nn.Dropout(dropout),
                                          nn.Linear(1024, 64),
                                          nn.ReLU(),
                                          nn.Linear(64, num_class))

    def forward(self, x):
        x1 = self.res_layer(x)
        x1_ = torch.cat((x1, x), dim=-1)
        out1 = self.norm(x1_)
        out1 = self.dropout(out1)

        x2 = self.res_layer(out1)
        x2_ = torch.cat((x2, out1), dim=-1)
        out2 = self.norm(x2_)
        out2 = self.dropout(out2)

        x3 = self.res_layer(out1)
        x3_ = torch.cat((x3, out2), dim=-1)
        out3 = self.norm(x3_)
        out3 = self.dropout(out3)

        out = self.classi_layer(out3)
        return out


if __name__ == '__main__':
    trail_input = torch.rand((2, 5), dtype=torch.float)

    model = DeepNet(input_size=5, num_class=1, dropout=0.5)
    out = model(trail_input)