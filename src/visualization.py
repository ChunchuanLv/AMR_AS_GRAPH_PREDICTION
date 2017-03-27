__author__ = 's1544871'

import torch.nn as nn
import torch
from torch.autograd import Variable
from npmt.Dict import *
import npmt
from npmt.Models import *
from npmt.data_iterator import *


from torch import cuda
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='train.py')

## Data options

parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from',default="model/valid_best.pt",
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")


## Optimization options

parser.add_argument('-total_size', type=int, default=1024,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=16,
                    help="Maximum batches of words in a sequence to run "
                         "the generator on in parallel. Higher is faster, but uses")

parser.add_argument('-gpus',  nargs='*',default=[0], type=int,
                    help="Use CUDA")

parser.add_argument('-log_interval', type=int, default=10,
                    help="Print stats at this interval.")
# parser.add_argument('-seed', type=int, default=3435,
#                     help="Seed for random initialization")

opt = parser.parse_args()

opt.cuda = len(opt.gpus)

from src.train import  my_loss,memoryEfficientLoss

def amr_seq_to_concept_str(amr_seq):
    con_seq = []
    for l in amr_seq:
        con_seq.append(l[1]+" "+l[3])
    con_seq.append(EOS_WORD)
    return con_seq

import matplotlib.colors as colors
class Visualizor(object):

    def visualizeOne(self,posterior,src,tgt,amr_ts):
     #   src = list(reversed(src))
    #    tgt =list(reversed(tgt))
        src.append(EOS_WORD)
        src_size = len(src)
        tgt_size = len(tgt)
        column_labels = tgt
        row_labels = src
        data = posterior.numpy()[:tgt_size,-src_size:]

        fig, ax = plt.subplots()
        heatmap = ax.pcolor(data)
        print (data.sum(1))
        print ("src",len(src))
        print ("amr",len(tgt))
        # put the major ticks at the middle of each cell, notice "reverse" use of dimension
        ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
        ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)

        ax.set_xticklabels(row_labels, minor=False)
        ax.set_yticklabels(column_labels, minor=False)

     #   plt.tick_params(axis='both', which='major', labelsize=7)
        plt.suptitle(amr_ts, fontsize=12, fontweight='bold')
        plt.show()
        return None
    def visualizeBatch(self, posterior,source):
        '''
        posterior:  tgt_len x batch x src_len
        '''
        posterior = posterior.transpose(0,1).contiguous().cpu().data
        print ("batch,tgt_len,src_len",posterior.size())
        source = [i for i in zip(*source)]
        snt_tokens = source[0]
        lemmas = source[1]
        amr_seqs = source[2]
        amr_ts = source[3]
        n = min(len(snt_tokens),5)
        for i in range(n):
            self.visualizeOne(posterior[i],snt_tokens[i],amr_seq_to_concept_str(amr_seqs[i]),amr_ts[i])



def main():
    word_dict = Dict("data/word_dict")
    lemma_dict = Dict("data/lemma_dict")
    pos_dict = Dict("data/pos_dict")
    ner_dict = Dict("data/ner_dict")
    rel_dict = Dict("data/rel_dict")
    concept_dict = Dict("data/concept_dict")
    category_dict = Dict("data/category_dict")


    word_dict.load()
    lemma_dict.load()
    pos_dict.load()
    ner_dict.load()
    rel_dict.load()
    concept_dict.load()
    category_dict.load()
    dicts = dict()
    dicts["word_dict"] = word_dict
    dicts["lemma_dict"] = lemma_dict
    dicts["pos_dict"] = pos_dict
    dicts["ner_dict"] = ner_dict
    dicts["rel_dict"] = rel_dict
    dicts["concept_dict"] = concept_dict
    dicts["category_dict"] = category_dict
    concept_ls = [id for id in concept_dict.idxToLabel.keys()]
    dicts["concept_ls"] = concept_ls

    checkpoint = torch.load(opt.train_from)
    
    optt = checkpoint['opt']
    if optt.cuda:
        cuda.set_device(optt.gpus[0])

    training_data = DataIterator([trainingFilesPath[1]],optt,volatile = True,dicts=dicts)

    dev_data = DataIterator([devFilesPath[1]],optt,volatile = True,dicts=dicts)

    model = checkpoint['model']
    model.eval()
    optim = checkpoint['optim']
    
    optt.start_epoch = checkpoint['epoch'] + 1

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)
    visualizor = Visualizor()


    x = training_data.getTranslation(1)
    srcBatch, tgtBatch, idBatch,mask,reBatch,source = x
    out,attns = model.forward(x)
    high_index,lemma_index = idBatch
    tgtBatch_t = tgtBatch[:,1:]
    high_index_t = high_index[1:]
    lemma_index_t = lemma_index[1:]
#       print ("trainModel",srcBatch.size())
    generator = model.generator
    loss,num_words,posterior  = memoryEfficientLoss(out,attns,tgtBatch_t, srcBatch,reBatch, high_index_t,lemma_index_t,generator,dicts,optt,True)


    visualizor.visualizeBatch(posterior,source)

if __name__ == "__main__":
    main()

