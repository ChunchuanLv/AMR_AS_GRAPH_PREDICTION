#!/usr/bin/env python3.6
# coding=utf-8
'''

Deep Learning Models for relation identification

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
from parser.modules.helper_module import mypack ,myunpack,MyPackedSequence,MyDoublePackedSequence,mydoubleunpack,mydoublepack,DoublePackedSequence,doubleunpack,data_dropout
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
from utility.constants import *



#sentence encoder for root identification
class RootSentenceEncoder(nn.Module):

    def __init__(self, opt, embs):
        self.layers = opt.root_enlayers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.txt_rnn_size % self.num_directions == 0
        self.hidden_size = opt.rel_rnn_size // self.num_directions
        inputSize =  embs["word_fix_lut"].embedding_dim + embs["lemma_lut"].embedding_dim\
                     +embs["pos_lut"].embedding_dim+embs["ner_lut"].embedding_dim

        super(RootSentenceEncoder, self).__init__()


        self.rnn =nn.LSTM(inputSize, self.hidden_size,
                        num_layers=self.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn,
                           batch_first=True)

        self.lemma_lut = embs["lemma_lut"]

        self.word_fix_lut  = embs["word_fix_lut"]

        self.pos_lut = embs["pos_lut"]


        self.ner_lut = embs["ner_lut"]

        self.alpha = opt.alpha
        if opt.cuda:
            self.rnn.cuda()



    def forward(self, packed_input,hidden=None):
    #input: pack(data x n_feature ,batch_size)
    #posterior: pack(data x src_len ,batch_size)
        assert isinstance(packed_input,PackedSequence)
        input = packed_input.data

        if self.alpha and self.training:
            input = data_dropout(input,self.alpha)

        word_fix_embed = self.word_fix_lut(input[:,TXT_WORD])
        lemma_emb = self.lemma_lut(input[:,TXT_LEMMA])
        pos_emb = self.pos_lut(input[:,TXT_POS])
        ner_emb = self.ner_lut(input[:,TXT_NER])

        emb = torch.cat([word_fix_embed,lemma_emb,pos_emb,ner_emb],1)#  data,embed

        emb = PackedSequence(emb, packed_input.batch_sizes)

        outputs = self.rnn(emb, hidden)[0]

        return  outputs

#combine amr node embedding and aligned sentence token embedding
class RootEncoder(nn.Module):

    def __init__(self, opt, embs):
        self.layers = opt.amr_enlayers
        #share hyper parameter with relation model
        self.size = opt.rel_dim
        inputSize = embs["cat_lut"].embedding_dim + embs["lemma_lut"].embedding_dim+opt.rel_rnn_size
        super(RootEncoder, self).__init__()

        self.cat_lut = embs["cat_lut"]

        self.lemma_lut  = embs["lemma_lut"]

        self.root = nn.Sequential(
            nn.Dropout(opt.dropout),
          nn.Linear(inputSize,self.size ),
            nn.ReLU()
        )


        self.alpha = opt.alpha
        if opt.cuda:
            self.cuda()

    def getEmb(self,indexes,src_enc):
        head_emb,lengths = [],[]
        src_enc = myunpack(*src_enc)  #  pre_amr_l/src_l  x batch x dim
        for i, index in enumerate(indexes):
            enc = src_enc[i]  #src_l x  dim
            head_emb.append(enc[index])  #var(amr_l  x dim)
            lengths.append(len(index))
        return mypack(head_emb,lengths)

    #input: all_data x n_feature, lengths
    #index: batch_size x var(amr_len)
    #src_enc   (batch x amr_len) x src_len x txt_rnn_size

    #head: batch   x var( amr_len x txt_rnn_size )

    #dep : batch x var( amr_len x amr_len x txt_rnn_size )

    #heads: [var(len),rel_dim]
    #deps: [var(len)**2,rel_dim]
    def forward(self, input, index,src_enc):
        assert isinstance(input, MyPackedSequence),input
        input,lengths = input
        if self.alpha and self.training:
            input = data_dropout(input,self.alpha)
        cat_embed = self.cat_lut(input[:,AMR_CAT])
        lemma_embed = self.lemma_lut(input[:,AMR_LE])

        amr_emb = torch.cat([cat_embed,lemma_embed],1)
    #    print (input,lengths)

        head_emb = self.getEmb(index,src_enc)  #packed, mydoublepacked


        root_emb = torch.cat([amr_emb,head_emb.data],1)
        root_emb = self.root(root_emb)

        return MyPackedSequence(root_emb,lengths)

#multi pass sentence encoder for relation identification
class RelSentenceEncoder(nn.Module):

    def __init__(self, opt, embs):
        self.layers = opt.rel_enlayers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.txt_rnn_size % self.num_directions == 0
        self.hidden_size = opt.rel_rnn_size // self.num_directions
        inputSize =  embs["word_fix_lut"].embedding_dim + embs["lemma_lut"].embedding_dim\
                     +embs["pos_lut"].embedding_dim+embs["ner_lut"].embedding_dim+1
        super(RelSentenceEncoder, self).__init__()


        self.rnn =nn.LSTM(inputSize, self.hidden_size,
                        num_layers=self.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn,
                           batch_first=True)   #first is for root

        self.lemma_lut = embs["lemma_lut"]

        self.word_fix_lut  = embs["word_fix_lut"]

        self.pos_lut = embs["pos_lut"]


        self.ner_lut = embs["ner_lut"]

        self.alpha = opt.alpha
        if opt.cuda:
            self.rnn.cuda()

    def posteriorIndictedEmb(self,embs,posterior):
        #real alignment is sent in as list of index
        #variational relaxed posterior is sent in as MyPackedSequence

        #out   (batch x amr_len) x src_len x (dim+1)
        embs,src_len = unpack(embs)

        if isinstance(posterior,MyPackedSequence):
       #     print ("posterior is packed")
            posterior = myunpack(*posterior)
            embs = embs.transpose(0,1)
            out = []
            lengths = []
            amr_len = [len(p) for p in posterior]
            for i,emb in enumerate(embs):
                expanded_emb = emb.unsqueeze(0).expand([amr_len[i]]+[i for i in emb.size()]) # amr_len x src_len x dim
                indicator = posterior[i].unsqueeze(2)  # amr_len x src_len x 1
                out.append(torch.cat([expanded_emb,indicator],2))  # amr_len x src_len x (dim+1)
                lengths = lengths + [src_len[i]]*amr_len[i]
            data = torch.cat(out,dim=0)

            return pack(data,lengths,batch_first=True),amr_len
        elif isinstance(posterior,list):
            embs = embs.transpose(0,1)
            src_l = embs.size(1)
            amr_len = [len(i) for i in posterior]
            out = []
            lengths = []
            for i,emb in enumerate(embs):
                amr_l = len(posterior[i])
                expanded_emb = emb.unsqueeze(0).expand([amr_l]+[i for i in emb.size()]) # amr_len x src_len x dim
                indicator = emb.data.new(amr_l,src_l).zero_()
                indicator.scatter_(1, posterior[i].data.unsqueeze(1), 1.0) # amr_len x src_len x 1
                indicator = Variable(indicator.unsqueeze(2))
                out.append(torch.cat([expanded_emb,indicator],2))  # amr_len x src_len x (dim+1)
                lengths = lengths + [src_len[i]]*amr_l
            data = torch.cat(out,dim=0)

            return pack(data,lengths,batch_first=True),amr_len


    def forward(self, packed_input, packed_posterior,hidden=None):
    #input: pack(data x n_feature ,batch_size)
    #posterior: pack(data x src_len ,batch_size)
        assert isinstance(packed_input,PackedSequence)
        input = packed_input.data

        if self.alpha and self.training:
            input = data_dropout(input,self.alpha)
        word_fix_embed = self.word_fix_lut(input[:,TXT_WORD])
        lemma_emb = self.lemma_lut(input[:,TXT_LEMMA])
        pos_emb = self.pos_lut(input[:,TXT_POS])
        ner_emb = self.ner_lut(input[:,TXT_NER])

        emb = torch.cat([word_fix_embed,lemma_emb,pos_emb,ner_emb],1)#  data,embed

        emb = PackedSequence(emb, packed_input.batch_sizes)
        poster_emb,amr_len = self.posteriorIndictedEmb(emb,packed_posterior)

        Outputs = self.rnn(poster_emb, hidden)[0]

        return  DoublePackedSequence(Outputs,amr_len,Outputs.data)


#combine amr node embedding and aligned sentence token embedding
class RelEncoder(nn.Module):

    def __init__(self, opt, embs):
        super(RelEncoder, self).__init__()

        self.layers = opt.amr_enlayers

        self.size = opt.rel_dim
        inputSize = embs["cat_lut"].embedding_dim + embs["lemma_lut"].embedding_dim+opt.rel_rnn_size

        self.head = nn.Sequential(
            nn.Dropout(opt.dropout),
          nn.Linear(inputSize,self.size )
        )

        self.dep = nn.Sequential(
            nn.Dropout(opt.dropout),
          nn.Linear(inputSize,self.size )
        )

        self.cat_lut = embs["cat_lut"]

        self.lemma_lut  = embs["lemma_lut"]
        self.alpha = opt.alpha

        if opt.cuda:
            self.cuda()

    def getEmb(self,indexes,src_enc):
        head_emb,dep_emb = [],[]
        src_enc,src_l = doubleunpack(src_enc)  # batch x var(amr_l x src_l x dim)
        length_pairs = []
        for i, index in enumerate(indexes):
            enc = src_enc[i]  #amr_l src_l dim
            dep_emb.append(enc.index_select(1,index))  #var(amr_l x amr_l x dim)
            head_index = index.unsqueeze(1).unsqueeze(2).expand(enc.size(0),1,enc.size(-1))
       #     print ("getEmb",enc.size(),dep_index.size(),head_index.size())
            head_emb.append(enc.gather(1,head_index).squeeze(1))  #var(amr_l  x dim)
            length_pairs.append([len(index),len(index)])
        return mypack(head_emb,[ls[0] for ls in length_pairs]),mydoublepack(dep_emb,length_pairs),length_pairs

    #input: all_data x n_feature, lengths
    #index: batch_size x var(amr_len)
    #src_enc   (batch x amr_len) x src_len x txt_rnn_size

    #head: batch   x var( amr_len x txt_rnn_size )

    #dep : batch x var( amr_len x amr_len x txt_rnn_size )

    #heads: [var(len),rel_dim]
    #deps: [var(len)**2,rel_dim]
    def forward(self, input, index,src_enc):
        assert isinstance(input, MyPackedSequence),input
        input,lengths = input
        if self.alpha and self.training:
            input = data_dropout(input,self.alpha)
        cat_embed = self.cat_lut(input[:,AMR_CAT])
        lemma_embed = self.lemma_lut(input[:,AMR_LE])

        amr_emb = torch.cat([cat_embed,lemma_embed],1)
    #    print (input,lengths)

        head_emb_t,dep_emb_t,length_pairs = self.getEmb(index,src_enc)  #packed, mydoublepacked


        head_emb = torch.cat([amr_emb,head_emb_t.data],1)

        dep_amr_emb_t = myunpack(*MyPackedSequence(amr_emb,lengths))
        dep_amr_emb = [ emb.unsqueeze(0).expand(emb.size(0),emb.size(0),emb.size(-1))      for emb in dep_amr_emb_t]

        mydouble_amr_emb = mydoublepack(dep_amr_emb,length_pairs)

    #    print ("rel_encoder",mydouble_amr_emb.data.size(),dep_emb_t.data.size())
        dep_emb = torch.cat([mydouble_amr_emb.data,dep_emb_t.data],-1)

       # emb_unpacked = myunpack(emb,lengths)

        head_packed = MyPackedSequence(self.head(head_emb),lengths) #  total,rel_dim
        head_amr_packed = MyPackedSequence(amr_emb,lengths) #  total,rel_dim

   #     print ("dep_emb",dep_emb.size())
        size = dep_emb.size()
        dep = self.dep(dep_emb.view(-1,size[-1])).view(size[0],size[1],-1)

        dep_packed  = MyDoublePackedSequence(MyPackedSequence(dep,mydouble_amr_emb[0][1]),mydouble_amr_emb[1],dep)

        return  head_amr_packed,head_packed,dep_packed  #,MyPackedSequence(emb,lengths)


class RelModel(nn.Module):
    def __init__(self, opt,embs):
        super(RelModel, self).__init__()
        self.root_encoder = RootEncoder(opt,embs)
        self.encoder = RelEncoder( opt, embs)
        self.generator = RelCalssifierBiLinear( opt, embs,embs["rel_lut"].num_embeddings)

        self.root = nn.Linear(opt.rel_dim,1)
        self.LogSoftmax = nn.LogSoftmax()


    def root_score(self,mypackedhead):
        heads = myunpack(*mypackedhead)
        output = []
        for head in heads:
            score = self.root(head).squeeze(1)
            output.append(self.LogSoftmax(score))
        return output

    def forward(self, srlBatch, index,src_enc,root_enc):
        mypacked_root_enc = self.root_encoder(srlBatch, index,root_enc) #with information from le cat enc
        roots = self.root_score(mypacked_root_enc)

        encoded= self.encoder(srlBatch, index,src_enc)
        score_packed = self.generator(*encoded)

        return score_packed,roots #,arg_logit_packed


class RelCalssifierBiLinear(nn.Module):

    def __init__(self, opt, embs,n_rel):
        super(RelCalssifierBiLinear, self).__init__()
        self.n_rel = n_rel
        self.cat_lut = embs["cat_lut"]
        self.inputSize = opt.rel_dim


        self.bilinear = nn.Sequential(nn.Dropout(opt.dropout),
                                  nn.Linear(self.inputSize,self.inputSize* self.n_rel))
        self.head_bias = nn.Sequential(nn.Dropout(opt.dropout),
                                   nn.Linear(self.inputSize,self.n_rel))
        self.dep_bias = nn.Sequential(nn.Dropout(opt.dropout),
                                      nn.Linear(self.inputSize,self.n_rel))
        self.bias = nn.Parameter(torch.normal(torch.zeros(self.n_rel)).cuda())


     #   self.lsm = nn.LogSoftmax()
        self.cat_lut = embs["cat_lut"]
        self.lemma_lut  = embs["lemma_lut"]
        if opt.cuda:
            self.cuda()

    def bilinearForParallel(self,inputs,length_pairs):
        output = []
        ls = []
        for i,input in enumerate(inputs):

            #head_t : amr_l x (  rel_dim x n_rel)
            #dep_t : amr_l x amr_l x rel_dim
            #head_bias : amr_l  x n_rel
            #dep_bias : amr_l  x   amr_l  x n_rel
            head_t,dep_t,head_bias,dep_bias = input
            l = len(head_t)
            ls.append(l)
            head_t = head_t.view(l,-1,self.n_rel)
            score =dep_t[:,:length_pairs[i][1]].bmm( head_t.view(l,-1,self.n_rel)).view(l,l,self.n_rel).transpose(0,1)

            dep_bias =  dep_bias[:,:length_pairs[i][1]]
            score = score + dep_bias

            score = score + head_bias.unsqueeze(1).expand_as(score)
            score = score+self.bias.unsqueeze(0).unsqueeze(1).expand_as(score)
            score = F.log_softmax(score.view(ls[-1]*ls[-1],self.n_rel)) # - score.exp().sum(2,keepdim=True).log().expand_as(score)
            
            output.append(score.view(ls[-1]*ls[-1],self.n_rel))
        return output,[l**2 for l in ls]


    def forward(self, _,heads,deps):
        '''heads.data: mypacked        amr_l x rel_dim
            deps.data: mydoublepacked     amr_l x amr_l x rel_dim
        '''
        heads_data = heads.data
        deps_data = deps.data

        head_bilinear_transformed = self.bilinear (heads_data)  #all_data x (    n_rel x inputsize)

        head_bias_unpacked = myunpack(self.head_bias(heads_data),heads.lengths) #[len x n_rel]

        size = deps_data.size()
        dep_bias =  self.dep_bias(deps_data.view(-1,size[-1])).view(size[0],size[1],-1)

        dep_bias_unpacked,length_pairs = mydoubleunpack(MyDoublePackedSequence(MyPackedSequence( dep_bias,deps[0][1]),deps[1],dep_bias) ) #[len x n_rel]

        bilinear_unpacked = myunpack(head_bilinear_transformed,heads.lengths)

        deps_unpacked,length_pairs = mydoubleunpack(deps)
        output,l = self.bilinearForParallel( zip(bilinear_unpacked,deps_unpacked,head_bias_unpacked,dep_bias_unpacked),length_pairs)
        myscore_packed = mypack(output,l)

      #  prob_packed = MyPackedSequence(myscore_packed.data,l)
        return myscore_packed