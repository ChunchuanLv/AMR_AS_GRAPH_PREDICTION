#!/usr/bin/env python3.6
# coding=utf-8
'''

Deep Learning Models for concept identification

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

import torch
import torch.nn as nn
from parser.modules.helper_module import data_dropout
from torch.nn.utils.rnn import PackedSequence
from utility.constants import *


class SentenceEncoder(nn.Module):
    def __init__(self, opt, embs):
        self.layers = opt.txt_enlayers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.txt_rnn_size % self.num_directions == 0
        self.hidden_size = opt.txt_rnn_size // self.num_directions
    #    inputSize = opt.word_dim*2 + opt.lemma_dim + opt.pos_dim +opt.ner_dim
        inputSize = embs["word_fix_lut"].embedding_dim +  embs["lemma_lut"].embedding_dim\
                    +embs["pos_lut"].embedding_dim + embs["ner_lut"].embedding_dim

        super(SentenceEncoder, self).__init__()
        self.rnn = nn.LSTM(inputSize, self.hidden_size,
                        num_layers=self.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)


        self.lemma_lut = embs["lemma_lut"]

        self.word_fix_lut  = embs["word_fix_lut"]


        self.pos_lut = embs["pos_lut"]

        self.ner_lut = embs["ner_lut"]

        self.drop_emb = nn.Dropout(opt.dropout)
        self.alpha = opt.alpha

        if opt.cuda:
            self.rnn.cuda()

    def forward(self, packed_input: PackedSequence,hidden=None):
    #input: pack(data x n_feature ,batch_size)
        input = packed_input.data
        if self.alpha and self.training:
            input = data_dropout(input,self.alpha)

        word_fix_embed = self.word_fix_lut(input[:,TXT_WORD])
        lemma_emb = self.lemma_lut(input[:,TXT_LEMMA])
        pos_emb = self.pos_lut(input[:,TXT_POS])
        ner_emb = self.ner_lut(input[:,TXT_NER])


        emb = self.drop_emb(torch.cat([lemma_emb,pos_emb,ner_emb],1))#  data,embed
        emb = torch.cat([word_fix_embed,emb],1)#  data,embed
        emb =  PackedSequence(emb, packed_input.batch_sizes)
        outputs, hidden_t = self.rnn(emb, hidden)
        return  outputs

class Concept_Classifier(nn.Module):

    def __init__(self, opt, embs):
        super(Concept_Classifier, self).__init__()
        self.txt_rnn_size = opt.txt_rnn_size

        self.n_cat = embs["cat_lut"].num_embeddings
        self.n_high = embs["high_lut"].num_embeddings
        self.n_aux = embs["aux_lut"].num_embeddings

        self.cat_score =nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(self.txt_rnn_size,self.n_cat,bias = opt.cat_bias))

        self.le_score =nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(self.txt_rnn_size,self.n_high+1,bias = opt.lemma_bias))

        self.ner_score =nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(self.txt_rnn_size,self.n_aux,bias = opt.cat_bias))

        self.t = 1
        self.sm =  nn.Softmax()
        if opt.cuda:
            self.cuda()



    def forward(self, src_enc ):
        '''
            src_enc: pack(data x txt_rnn_size ,batch_size)
           src_le:  pack(data x 1 ,batch_size)

           out:  (datax n_cat, batch_size),    (data x n_high+1,batch_size)
        '''

        assert isinstance(src_enc,PackedSequence)


     #   high_embs = self.high_lut.weight.expand(le_score.size(0),self.n_high,self.dim)
      #  le_self_embs = self.lemma_lut(src_le.data).unsqueeze(1)
      #  le_emb = torch.cat([high_embs,le_self_embs],dim=1) #data x high+1 x dim

        pre_enc =src_enc.data

        cat_score = self.cat_score(pre_enc) #  n_data x n_cat
        ner_score = self.ner_score(pre_enc)#  n_data x n_cat
        le_score = self.le_score (src_enc.data)
        le_prob = self.sm(le_score)
        cat_prob = self.sm(cat_score)
        ner_prob = self.sm(ner_score)
        batch_sizes = src_enc.batch_sizes
        return   PackedSequence(cat_prob,batch_sizes),PackedSequence(le_prob,batch_sizes),PackedSequence(ner_prob,batch_sizes)

class ConceptIdentifier(nn.Module):
    #could share encoder with other model
    def __init__(self, opt,embs,encoder = None):
        super(ConceptIdentifier, self).__init__()
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = SentenceEncoder( opt, embs)
        self.generator = Concept_Classifier( opt, embs)


    def forward(self, srcBatch):
        src_enc = self.encoder(srcBatch)
        probBatch = self.generator(src_enc)
        return probBatch,src_enc
