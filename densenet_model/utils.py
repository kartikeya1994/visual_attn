import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

class BoxCar(nn.Sequential):
    
    def __init__(self, ch, dim1, dim2):
        super(BoxCar, self).__init__()
        f1 = torch.autograd.Variable(torch.from_numpy(np.arange(dim1)).view(1, 1, 1, dim1, -1))
        f2 = torch.autograd.Variable(torch.from_numpy(np.arange(dim2)).view(1, 1, 1, 1, -1))
        z1 = torch.autograd.Variable(torch.zeros(ch*dim1).long().view(1, 1, ch, dim1, -1))
        z2 = torch.autograd.Variable(torch.zeros(ch*dim2).long().view(1, 1, ch, 1, -1))
        self.f1 = f1 + z1
        self.f2 = f2 + z2

    def logistic(x, k):
        return 1.0/(1 + torch.exp(-k * x))
    
    def forward(self, x, m):
        '''
            x -  s x ch x dim1 x dim2
            m - s x g x 4
        '''
        
        s = x.size(0)
        g = m.size(1)
        M = []
        for i in range(g):
            m1 = logistic((self.f1 - m[:, i, 0].contiguous().view(s, 1, 1, 1, -1)).float(), k)
            m2 = logistic((self.f1 - m[:, i, 2].contiguous().view(s, 1, 1, 1, -1)).float(), k)
            m3 = logistic((self.f2 - m[:, i, 1].contiguous().view(s, 1, 1, 1, -1)).float(), k)
            m4 = logistic((self.f2 - m[:, i, 3].contiguous().view(s, 1, 1, 1, -1)).float(), k)
            v = (m1-m2)*(m3-m4)
            M.append(v)

        M = torch.cat(M, 1)
        return x.view(s, 1, ch, dim1, -1) * M
