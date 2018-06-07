#!/usr/bin/env python3.6
# coding=utf-8
'''

Iterating over data set

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''
from utility.constants import *
from utility.data_helper import *
import torch
from torch.autograd import Variable
import math
from torch.nn.utils.rnn import PackedSequence
from parser.modules.helper_module import MyPackedSequence
from torch.nn.utils.rnn import pack_padded_sequence as pack
import re
end= re.compile(".txt\_[a-z]*")
def rel_to_batch(rel_batch_p,rel_index_batch_p,data_iterator,dicts):
    lemma_dict,category_dict = dicts["lemma_dict"], dicts["category_dict"]

    data = [torch.LongTensor([[category_dict[uni.cat],lemma_dict[uni.le],0] for uni in uni_seq]) for uni_seq in rel_batch_p ]
    rel_index = [torch.LongTensor(index) for index in rel_index_batch_p]

    rel_batch,rel_index_batch,rel_lengths = data_iterator._batchify_rel_concept(data,rel_index)
    return  MyPackedSequence(rel_batch,rel_lengths),rel_index_batch

class DataIterator(object):

    def __init__(self, filePathes,opt,rel_dict,volatile = False ,all_data = None):
        self.cuda = opt.gpus[0] != -1
        self.volatile = volatile
        self.rel_dict = rel_dict
        self.all = []
        self.opt = opt
       #     break
            
 #       self.all = sorted(self.all, key=lambda x: x[0])
        self.src = []
        self.tgt = []
        self.align_index = []
        self.rel_seq = []
        self.rel_index = []
        self.rel_mat = []
        self.root = []
        self.src_source = []
        self.tgt_source = []
        self.rel_tgt = []
        if all_data:
            for data in all_data:
                self.read_sentence(data)
            self.batchSize = len(all_data)
            self.numBatches = 1
        else:

            for filepath in filePathes:
                n = self.readFile(filepath)
            self.batchSize = opt.batch_size
            self.numBatches = math.ceil(len(self.src)/self.batchSize)

        self.source_only = len(self.root) == 0

    def read_sentence(self,data):
        def role_mat_to_sparse(role_mat,rel_dict):
            index =[]
            value = []
            for i,role_list in enumerate(role_mat):
                for role_index in role_list:
                    if role_index[0] in rel_dict:
                        index.append([i,role_index[1]])
                        value.append(rel_dict[role_index[0]])
            size = torch.Size([len(role_mat),len(role_mat)])
            v = torch.LongTensor(value)
            if len(v) == 0:
                i = torch.LongTensor([[0,0]]).t()
                v = torch.LongTensor([0])
                return torch.sparse.LongTensor(i,v,size)

            i = torch.LongTensor(index).t()
            return torch.sparse.LongTensor(i,v,size)

        #src: length x n_feature

        self.src.append(torch.LongTensor([data["snt_id"],data["lemma_id"],data["pos_id"],data["ner_id"]]).t().contiguous())

        #source

        self.src_source.append([data["tok"],data["lem"],data["pos"],data["ner"]])

        #tgt: length x n_feature
    #    print (data["amr_id"])
        if "amr_id" in data:
            self.tgt.append(torch.LongTensor(data["amr_id"]))  # lemma,cat, lemma_sense,ner,is_high
            self.align_index.append(data["index"])

            amrl = len(data["amr_id"])
            for i in data["amr_rel_index"]:
                assert i <amrl,data
            #rel
            self.rel_seq.append(torch.LongTensor(data["amr_rel_id"]))  # lemma,cat, lemma_sense
            self.rel_index.append(torch.LongTensor(data["amr_rel_index"]))
            mats = role_mat_to_sparse(data["roles_mat"], self.rel_dict)
            self.rel_mat.append(mats)  #role, index
            self.root.append(data["root"])  #role, index

            #source

            self.tgt_source.append([data["rel_seq"],data["rel_triples"],data["convertedl_seq"],data["amr_seq"],data["amr_t"]])





    def readFile(self,filepath):
        print ("reading "+filepath)
        data_file = Pickle_Helper(filepath)

        all_data = data_file.load()["data"]
        for data in all_data:
       #     print (data)
            self.read_sentence(data)
        print(("done reading "+filepath+", "+str( len(all_data))+" sentences processed"))
        return len(all_data)

    #align_index: batch_size x var(tgt_len) x []
    #out : batch_size x tgt_len x src_len
    def _batchify_align(self, align_index,max_len):
        out = torch.ByteTensor(len(align_index),max_len,max_len).fill_(0)
        for i in range(len(align_index)):
            for j in range(len(align_index[i])):
                if align_index[i][j][0] == -1:
                    out[i][j][:align_index[i][j][1]].fill_(1)
                else:
                    for k in align_index[i][j]:
                        out[i][j][k] = 1
            for j in range(len(align_index[i]),max_len):   #for padding
                out[i][j][len(align_index[i]):].fill_(1)
        return out

    #rel_seq: batch_size x var(len) x n_feature
    #rel_index: batch_size x var(len)

    #out : all_data x n_feature
    #out_index: batch_size x var(len)
    #lengths : batch_size
    def _batchify_rel_concept(self, data,rel_index ):
        lengths = [len(x) for x in data]
        for l in lengths:
            assert l >0, (data,rel_index)
        second = max([x.size(1) for x in data])
        total = sum(lengths)
        out = data[0].new(total, second)
        out_index = []
        current = 0
        for i in range(len(data)):
            data_t = data[i].clone()
            out.narrow(0, current, lengths[i]).copy_(data_t)
            index_t = rel_index[i].clone()
            if self.cuda:
                index_t = index_t.cuda()
            out_index.append(Variable(index_t,volatile=self.volatile,requires_grad = False))
          #  out_index.append(index_t)
            current += lengths[i]
        out = Variable(out,volatile=self.volatile,requires_grad = False)

        if self.cuda:
            out = out.cuda()
        return out,out_index,lengths


    #rel_mat: batch_size x var(len) x var(len)
    #rel_index: batch_size x var(len)

    #out :  (batch_size x var(len) x var(len))
    def _batchify_rel_roles(self, all_data ):
        length_squares = [x.size(0)**2 for x in all_data]
        total = sum(length_squares)
        out = torch.LongTensor(total)
        current = 0
        for i in range(len(all_data)):
            data_t = all_data[i].to_dense().clone().view(-1)
            out.narrow(0, current, length_squares[i]).copy_(data_t)
            current += length_squares[i]

        out = Variable(out,volatile=self.volatile,requires_grad = False)
        if self.cuda:
            out = out.cuda()

        return out,length_squares


    #data: batch_size x var(len) x n_feature
    #out : batch_size x tgt_len x n_feature
    def _batchify_tgt(self, data,max_src ):
        lengths = [x.size(0) for x in data]
        max_length = max(max(x.size(0) for x in data),max_src)   #if y, we need max_x
        out = data[0].new(len(data), max_length,data[0].size(1)).fill_(PAD)
        for i in range(len(data)):
            data_t = data[i].clone()
            data_length = data[i].size(0)
            out[i].narrow(0, 0, data_length).copy_(data_t)
        return out

    #data: batch_size x var(len) x n_feature
    #out : batch_size x src_len x n_feature
    def _batchify_src(self, data,max_length ):
        out = data[0].new(len(data), max_length,data[0].size(1)).fill_(PAD)

        for i in range(len(data)):
            data_t = data[i].clone()
            data_length = data[i].size(0)
            out[i].narrow(0, 0, data_length).copy_(data_t)

        return out

    def getLengths(self,index):
        src_data = self.src[index*self.batchSize:(index+1)*self.batchSize]
        src_lengths = [x.size(0) for x in src_data]
        if  self.source_only:
            return src_lengths,max(src_lengths)

        tgt_data = self.tgt[index*self.batchSize:(index+1)*self.batchSize]
        tgt_lengths = [x.size(0) for x in tgt_data]
        lengths = []
        for i,l in enumerate(src_lengths):
            lengths.append(max(l,tgt_lengths[i]))
        return lengths,max(lengths)

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        lengths,max_len = self.getLengths(index )
        def wrap(b,l ):
            #batch, len, feature
            if b is None:
                return b
            b = torch.stack(b, 0).transpose(0,1).contiguous()
            if self.cuda:
                b = b.cuda()
            packed =  pack(b,list(l))
            return PackedSequence(Variable(packed[0], volatile=self.volatile,requires_grad = False),packed[1])

        def wrap_align(b,l ):
            #batch, len_tgt, len_src
            if b is None:
                return b
            b = torch.stack(b, 0).transpose(0,1).contiguous().float()
            if self.cuda:
                b = b.cuda()
            packed =  pack(b,list(l))
            return PackedSequence(Variable(packed[0], volatile=self.volatile,requires_grad = False),packed[1])

        srcBatch = self._batchify_src(
            self.src[index*self.batchSize:(index+1)*self.batchSize],max_len)

        if self.source_only:
            src_sourceBatch = self.src_source[index*self.batchSize:(index+1)*self.batchSize]

            batch = zip( srcBatch,src_sourceBatch)
            lengths,max_len = self.getLengths(index )
            order_data =    sorted(list(enumerate(list(zip(batch, lengths)))),key = lambda x:-x[1][1])
            order,data = zip(*order_data)
            batch, lengths = zip(*data)
            srcBatch,src_sourceBatch = zip(*batch)
            return order,wrap(srcBatch,lengths),src_sourceBatch

        else:
            tgtBatch = self._batchify_tgt(
                    self.tgt[index*self.batchSize:(index+1)*self.batchSize],max_len)
            alignBatch = self._batchify_align(
                    self.align_index[index*self.batchSize:(index+1)*self.batchSize],max_len)

            rel_seq_pre = self.rel_seq[index*self.batchSize:(index+1)*self.batchSize]
            rel_index_pre = self.rel_index[index*self.batchSize:(index+1)*self.batchSize]
            rel_role_pre = self.rel_mat[index*self.batchSize:(index+1)*self.batchSize]

         #   roots = Variable(torch.IntTensor(self.root[index*self.batchSize:(index+1)*self.batchSize]),volatile = True)
            roots =self.root[index*self.batchSize:(index+1)*self.batchSize]

            src_sourceBatch = self.src_source[index*self.batchSize:(index+1)*self.batchSize]
            tgt_sourceBatch = self.tgt_source[index*self.batchSize:(index+1)*self.batchSize]
            sourceBatch = [  src_s +tgt_s for src_s,tgt_s in zip(src_sourceBatch,tgt_sourceBatch)]
            # within batch sorting by decreasing length for variable length rnns
            indices = range(len(srcBatch))

            batch = zip(indices, srcBatch ,tgtBatch,alignBatch,rel_seq_pre,rel_index_pre,rel_role_pre,sourceBatch)
            order_data =    sorted(list(enumerate(list(zip(batch, lengths)))),key = lambda x:-x[1][1])
            order,data = zip(*order_data)
            batch, lengths = zip(*data)
            indices, srcBatch,tgtBatch,alignBatch ,rel_seq_pre,rel_index_pre,rel_role_pre,sourceBatch= zip(*batch)

            rel_batch,rel_index_batch,rel_lengths = self._batchify_rel_concept(rel_seq_pre,rel_index_pre)
            rel_roles,length_squares = self._batchify_rel_roles(rel_role_pre)


    #,wrap(charBatch))
            return order,wrap(srcBatch,lengths), wrap(tgtBatch,lengths), wrap_align(alignBatch,lengths),\
                   MyPackedSequence(rel_batch,rel_lengths),rel_index_batch,MyPackedSequence(rel_roles,length_squares),roots,sourceBatch

    def __len__(self):
        return self.numBatches


    def shuffle(self):
    #    if True: return
        if self.source_only: #if data set if for testing
            data = list(zip(self.src,self.src_source))
            self.src,self.src_source = zip(*[data[i] for i in torch.randperm(len(data))])
        else:
            data = list(zip(self.src, self.tgt,self.align_index,self.rel_seq,self.rel_index,self.rel_mat,self.root,self.src_source,self.tgt_source))
            self.src, self.tgt,self.align_index,self.rel_seq,self.rel_index,self.rel_mat,self.root,self.src_source,self.tgt_source = zip(*[data[i] for i in torch.randperm(len(data))])

