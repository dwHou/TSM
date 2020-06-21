import torch
import torch.nn as nn
import torch.nn.init as init

'''
def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return RES(args, dilated.dilated_conv)
    else:
        return RES(args)
'''

class TemporalShift(nn.Module):
    def __init__(self, n_segment=5, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        # print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return x

    @staticmethod
    def shift(x, n_segment, fold_div, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left [b, t-1, c//8, h, w]
            out[:, -1, :fold] = x[:, 0, :fold] # zero padding → circulant padding
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right [b, t-1, c//8, h, w]
            out[:, 0, fold: 2 * fold] = x[:, -1, fold: 2 * fold] # zero padding → circulant padding
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift [b, t, rest_c, h, w]

        return out.view(nt, c, h, w)
    

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ResShiftBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True)):

        super(ResShiftBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(TemporalShift(n_segment=5, n_div=8, inplace=False))
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        # x_shift = self.tsm(x)
        res = self.body(x)
        res += x

        return res
    
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True)):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res    


class RES(nn.Module):
    def __init__(self, n_resblock, n_feats, res_scale, conv=default_conv):
        super(RES, self).__init__()

        kernel_size = 3 
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_body1 = [
            ResShiftBlock(
            conv, n_feats, kernel_size, act=act) for _ in range(1)
        ]
        m_body1.append(conv(n_feats, n_feats, kernel_size))
        
        m_body2 = [
            ResBlock(
            conv, n_feats, kernel_size, act=act) for _ in range(n_resblock)
        ]
        m_body2.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            nn.Conv2d(
                n_feats, 3, kernel_size,
                padding=(kernel_size//2)
            )
        ]
        self.head = nn.Sequential(*m_head)
        self.Encoder = nn.Sequential(*m_body1)
        self.trans = conv(5*n_feats, n_feats, kernel_size)
        self.Decoder = nn.Sequential(*m_body2)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        B, T, C, H, W = x.size()

        x_center = x[:, T//2, :, :, :].contiguous()

        x = x.view(-1, C, H, W)
        
        res = self.head(x)
        
        res = self.Encoder(res)
        
        res = res.view(B, -1, H, W)
        res = self.trans(res)
        
        res = self.Decoder(res)
        
        res = self.tail(res)
        
        y = res + x_center
        

        return y
