"""
Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.
        H_1 H_2 H_3 ... H_n
          q   q   q       q
            |  |   |       |
              \ |   |      /
                      .....
                  \   |  /
                          a
Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.
    The full def is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.:
"""

import torch
import torch.nn as nn
import math

_INF = float('inf')

class GlobalAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_in.weight.data.normal_(0,0.707/dim)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim*2, dim, bias=False)
        self.linear_out.weight.data.normal_(0,1.0/dim)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -_INF)
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        contextCombined = torch.cat((weightedContext, input), 1)

        contextOutput = self.tanh(self.linear_out(contextCombined))

        return contextOutput, attn
    
    
class LocalAttention(nn.Module):
    def __init__(self, dim):
        super(LocalAttention, self).__init__()
        self.dim = dim
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim*2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None
        self.flat = False

    def applyMask(self, mask):
        self.mask = mask
    def setFlat(self,flat):
        self.flat = flat
    #contextOuput: batch x sourceL x dim
    def forward(self, input, context):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        targetT = self.linear_in(input).unsqueeze(2)  # batch x dim +1
        
        # Get attention
        attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
      #  del targetT
        if self.mask is not None:
            if self.flat:
                attn.data.fill_(0)
            attn.data.masked_fill_(self.mask, -_INF)
        attn = self.sm(attn)   # batch x sourceL
        input_expanded = input.view(input.size(0),1,input.size(1)).expand(input.size(0),attn.size(1),input.size(1))

        contextCombined = torch.cat((context, input_expanded), 2).contiguous() # batch x sourceL x dim*2

        contextOutput = self.tanh(self.linear_out(contextCombined.view(contextCombined.size(0)*contextCombined.size(1)\
                                                                       ,contextCombined.size(2))))

        return contextOutput.view(input.size(0),attn.size(1),self.dim), attn # batch x sourceL x dim      batch x sourceL x dim