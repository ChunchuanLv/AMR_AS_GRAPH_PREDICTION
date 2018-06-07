import math,torch
import torch.optim as optim
import numpy as np
class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr,weight_decay = 0)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr,weight_decay = 0)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr,weight_decay = 0)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, betas=[0.9,0.9],lr=self.lr,weight_decay = 0)
        elif self.method == "RMSprop":
            self.optimizer = optim.RMSprop(self.params, lr=self.lr, weight_decay=0)

        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, max_grad_norm, lr_decay=1, start_decay_at=None, weight_decay=0,perturb = 0):
        self.params = list(params)  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self.weight_decay =  weight_decay
        self.weight_shirnk = 1.0 -weight_decay
        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        for param in self.params:
            grad_norm += math.pow(param.grad.data.norm(), 2)

        grad_norm = math.sqrt(grad_norm)
        shrinkage = self.max_grad_norm / grad_norm
        nan_size = []
        fine = []
        for param in self.params:
            if shrinkage < 1:
                 param.grad.data.mul_(shrinkage)
        #    assert not np.isnan(np.sum(param.data.cpu().numpy())),("befotr optim\n",param)
         #   if  np.isnan(np.sum(param.grad.data.cpu().numpy())):
        #        nan_size.append(param.grad.size())
        #    else: fine.append(param.grad.size())
        if len(nan_size) > 0:
            print ("befotr optim grad explodes, abandon update, still weight_decay\n",fine)
        self.optimizer.step()
        for param in self.params:
            assert not np.isnan(np.sum(param.data.cpu().numpy())),("befotr shrink\n",param)
            param.data.mul_(self.weight_shirnk) #+ torch.normal(0,1e-3*torch.ones(param.data.size()).cuda())
            assert not np.isnan(np.sum(param.data.cpu().numpy())),("after shrink\n",param)
        return grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_ppl = ppl

        self._makeOptimizer()

