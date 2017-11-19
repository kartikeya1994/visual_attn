import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import numpy as np
from torch.autograd import Variable
class BoxCar(nn.Module):
    
    def __init__(self, ch=3, dim1=299, dim2=299, k=100000, use_gpu=True):
        super(BoxCar, self).__init__()
        f1 = torch.from_numpy(np.arange(dim1)).view(1, 1, 1, dim1, -1)
        f2 = torch.from_numpy(np.arange(dim2)).view(1, 1, 1, 1, -1)
        z1 = torch.zeros(ch*dim1).long().view(1, 1, ch, dim1, -1)
        z2 = torch.zeros(ch*dim2).long().view(1, 1, ch, 1, -1)
        
        if use_gpu:
            f1 = Variable(f1.cuda())
            f2 = Variable(f2.cuda())
            z1 = Variable(z1.cuda())
            z2 = Variable(z2.cuda())
        else:
            f1 = Variable(f1)
            f2 = Variable(f2)
            z1 = Variable(z1)
            z2 = Variable(z2)

        self.f1 = f1 + z1
        self.f2 = f2 + z2
        self.k = k
        self.ch = ch
        self.dim1 = dim1
        self.dim2 = dim2

    def logistic(self, x):
        return 1.0/(1 + torch.exp(-self.k * x))
    
    def forward(self, x, m):
        '''
            x -  s x ch x dim1 x dim2
            m - s x g x 4
            returns - s x g x ch x dim1 x dim2
        '''
        
        s = x.size(0)
        g = m.size(1)
        M = []
        for i in range(g):
            m1 = self.logistic((self.f1 - m[:, i, 0].contiguous().view(s, 1, 1, 1, -1)).float())
            m2 = self.logistic((self.f1 - m[:, i, 2].contiguous().view(s, 1, 1, 1, -1)).float())
            m3 = self.logistic((self.f2 - m[:, i, 1].contiguous().view(s, 1, 1, 1, -1)).float())
            m4 = self.logistic((self.f2 - m[:, i, 3].contiguous().view(s, 1, 1, 1, -1)).float())
            v = (m1-m2)*(m3-m4)
            M.append(v)

        M = torch.cat(M, 1)
        return x.view(s, 1, self.ch, self.dim1, -1) * M


class Upsampler(nn.Module):
    def __init__(self, target_dim=(299,299), mode='bilinear'):

        super(Upsampler, self).__init__()
        self.h = target_dim[0]
        self.w = target_dim[1]

        self.upsampler = torch.nn.Upsample(size=target_dim, mode=mode)
        
    def img_crop(self, x, tl_x, tl_y, br_x, br_y, target_size=(299,299)):
        """
        Takes tensor of dimension x: (3, 299, 299) and
        f: (s, 4) containing tl_x, tl_y, br_x, br_y in that
        order. Returns upsampled crops
        """
        # note that the following step is not 
        # a part of the network, taking values
        # out of the tensor here
        tl_x, tl_y, br_x, br_y = int(tl_x.data[0]), int(tl_y.data[0]), int(br_x.data[0]), int(br_y.data[0])
        #cropped = img_set_zero(x, tl_x, tl_y, br_x, br_y)
        #cropped = cropped[:,tl_x:br_x,tl_y:br_y].contiguous()
        cropped = x[:,tl_x:br_x,tl_y:br_y].contiguous()
        cropped = cropped.view(1, 3, cropped.size(1), cropped.size(2))
        bi = torch.nn.Upsample(size=(299,299), mode='bilinear')
        upped = bi(cropped).view(3, 299, 299)
        return upped

    def img_crops(self, x, f):
        """
        x: (3, 299, 299)
        f: (g, 4) tl_x, tl_y, br_x, br_y
        returns cropped and upsampled same as x.size
        """
        out = []
        for f_i in torch.unbind(f):
            out.append(self.img_crop(x, f_i[0], f_i[1], f_i[2], f_i[3]))
        out = torch.stack(out, 0)
        return out

    def imgs_crops(self, x, f):
        """
        x: (s, 3, 299, 299)
        f: (s, g, 4) tl_x, tl_y, br_x, br_y
        returns cropped and upsampled same as x.size
        """
        out = []
        for i,x_i in enumerate(torch.unbind(x)):
            out.append(self.img_crops(x_i, f[i]))
        out = torch.stack(out, 0)
        return out

    def forward(self, x, f):
        return self.imgs_crops(x, f)

