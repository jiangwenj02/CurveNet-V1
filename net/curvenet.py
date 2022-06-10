import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class CurveNet(nn.Module):
    def __init__(self, factor=5, input_dim=295, actf = 'sigmoid', bias = 0, num_cls=10, hiddim=64):
        super(CurveNet, self).__init__()
        print('input_dim', input_dim)
        self.factor = factor
        self.input_dim = input_dim
        self.before_linear1 = nn.Linear(input_dim, hiddim * 2)
        self.before_linear2 = nn.Linear(hiddim * 2, hiddim * 2)
        self.before_linear3 = nn.Linear(hiddim * 2, hiddim)

        self.cls_emb = nn.Embedding(num_cls, hiddim)
        self.hiddim = hiddim

        # self.flag_emb = nn.Embedding(2, hiddim)

        self.linear1 = nn.Linear(hiddim, hiddim * 2)
        self.linear2 = nn.Linear(hiddim * 2, hiddim)
        self.linear3 = nn.Linear(hiddim, 1)

        self.apply(weights_init)
        self.actf_str = actf
        self.init_weight()
        self.bias = bias
        if actf == 'sigmoid':
            self.actf = torch.sigmoid
        elif actf == 'relu':
            self.actf = torch.relu
        elif actf == 'none':
            self.actf = None
        
    def init_weight(self):
        nn.init.xavier_uniform_(self.cls_emb.weight)
        # nn.init.xavier_uniform_(self.flag_emb.weight)
        nn.init.xavier_normal_(self.before_linear1.weight)
        nn.init.xavier_normal_(self.before_linear2.weight)
        nn.init.xavier_normal_(self.before_linear3.weight)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.xavier_normal_(self.linear3.weight)

        self.linear2.bias.data.zero_()
        self.linear3.bias.data.zero_()
        self.before_linear1.bias.data.zero_()
        self.before_linear2.bias.data.zero_()
        self.before_linear3.bias.data.zero_()
        if self.actf_str == 'sigmoid':
            init.constant_(self.linear3.bias, 4.5)
        elif self.actf_str == 'relu' or self.actf_str == 'none':
            init.constant_(self.linear3.bias, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, y):
        out = self.before_linear1(x)
        out = F.relu(out)
        out = self.before_linear2(out)
        out = F.relu(out)
        out = self.before_linear3(out)
        out = out.unsqueeze(1)

        out = F.adaptive_avg_pool1d(out, self.hiddim)
        out = F.relu(out)
        out = out.view(out.size(0), -1)
        
        y_emb = self.cls_emb(y)

        out = out + y_emb

        out = out + y_emb

        out = self.linear1(out)
        out = torch.relu(out)
        out = self.linear2(out)
        out = torch.relu(out)
        out = self.linear3(out)
        
        if self.actf_str == 'none':
            return self.factor * out
        return self.factor * self.actf(out) + self.bias
