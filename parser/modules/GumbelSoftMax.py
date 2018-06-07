#!/usr/bin/env python3.6
# coding=utf-8
'''

Helper functions regarding gumbel noise

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence

eps = 1e-8
def sample_gumbel(input):
    noise = torch.rand(input.size()).type_as(input.data)
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return Variable(noise,requires_grad=False)


def gumbel_noise_sample(input,temperature = 1):
    noise = sample_gumbel(input)
    x = (input + noise) / temperature
    return x.view_as(input)


import numpy as np

def sink_horn(input,k = 5,t = 1,batch_first = False):
    def sink_horn_data(x,lengths):
        assert not np.isnan(np.sum(x.data.cpu().numpy())),("start x\n",x.data)
        over_flow = x-80*t
        x = x.clamp(max=80*t)+F.tanh(over_flow)*(over_flow>0).float()
        x = torch.exp(x/t)
        assert not np.isnan(np.sum(x.data.cpu().numpy())),("exp x\n",x.data)
        musks = torch.zeros(x.size())
        for i,l in enumerate(lengths):
            musks[:l,i,:l] = 1
        musks = Variable(musks,requires_grad=False).type_as(x)
        x = x*musks+eps
        for i in range(0,k):
            x = x / x.sum(0,keepdim=True).expand_as(x)
            x = x*musks+eps
            x = x / x.sum(2,keepdim=True).expand_as(x)
            x = x*musks+eps

        assert not np.isnan(np.sum(x.data.cpu().numpy())),("end x\n",x.data)
        return x
    if isinstance(input,PackedSequence):
        data,l = unpack(input,batch_first=batch_first)
        data = sink_horn_data(data,l)
        return pack(data,l,batch_first)
    else:
        return sink_horn_data(*input)


def renormalize(input,t=1):

    x = ((input+eps).log() ) / t
    x = F.softmax(x)
    return x.view_as(input)

