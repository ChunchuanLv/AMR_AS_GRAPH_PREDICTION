#!/usr/bin/env python3.6
# coding=utf-8
'''

Deep Learning Models for variational inference of alignment.
Posterior , LikeliHood helps computing posterior weighted likelihood regarding relaxation.

Also the whole AMR model is combined here.

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

import numpy as np
from parser.models.ConceptModel import *
from parser.models.MultiPassRelModel import *

from parser.modules.GumbelSoftMax import renormalize,sink_horn,gumbel_noise_sample
from parser.modules.helper_module import doublepack

from copy import deepcopy

#Encoding linearized AMR concepts for vartiaonal alignment model
class AmrEncoder(nn.Module):

    def __init__(self, opt, embs):
        self.layers = opt.amr_enlayers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.amr_rnn_size % self.num_directions == 0
        self.hidden_size = opt.amr_rnn_size // self.num_directions
        inputSize = embs["cat_lut"].embedding_dim +  embs["lemma_lut"].embedding_dim
        super(AmrEncoder, self).__init__()

        self.rnn = nn.LSTM(inputSize, self.hidden_size,
                        num_layers=opt.amr_enlayers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)
        self.cat_lut = embs["cat_lut"]

        self.lemma_lut  = embs["lemma_lut"]



        self.alpha = opt.alpha  #unk with alpha
        if opt.cuda:
            self.cuda()

    #input:len,  batch, n_feature
    #output: len, batch, hidden_size * num_directions
    def forward(self, packed_input, hidden=None):
        assert isinstance(packed_input,PackedSequence)
        input = packed_input.data

        if self.alpha and self.training:
            input = data_dropout(input,self.alpha)

        cat_embed = self.cat_lut(input[:,AMR_CAT])
        lemma_embed = self.lemma_lut(input[:,AMR_LE])

        emb = torch.cat([cat_embed,lemma_embed],1) #  len,batch,embed
        emb = PackedSequence(emb, packed_input.batch_sizes)
        outputs, hidden_t = self.rnn(emb, hidden)
        return  outputs, hidden_t

#Model to compute relaxed posteior
# we constraint alignment if copying mechanism can be used
class Posterior(nn.Module):
    def __init__(self,opt):
        super(Posterior, self).__init__()
        self.txt_rnn_size = opt.txt_rnn_size
        self.amr_rnn_size = opt.amr_rnn_size
        self.jamr = opt.jamr
        if self.jamr : #if use fixed alignment, then no need for variational model
            return
        self.transform = nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(self.txt_rnn_size,self.amr_rnn_size,bias = opt.lemma_bias))
        self.sm = nn.Softmax()
        self.sink = opt.sink
        self.sink_t = opt.sink_t
        if opt.cuda:
            self.cuda()

    def forward(self,src_enc,amr_enc,aligns):

        '''src_enc:   src_len x  batch x txt_rnn_size, src_l
           amr_enc: amr_len x  batch x  amr_rnn_size, amr_l
            aligns:  amr_len x  batch x src_len , amr_l


            posterior: amr_len x  batch x src_len , amr_l
        '''
        if self.jamr :
            return aligns,aligns,0
        src_enc,amr_enc,aligns  =unpack(src_enc),unpack(amr_enc),unpack(aligns)

        src_enc = src_enc[0]
        amr_enc = amr_enc[0]
        lengths = aligns[1]
        aligns = aligns[0]
        assert not np.isnan(np.sum(src_enc.data.cpu().numpy())),("src_enc \n",src_enc)
        assert not np.isnan(np.sum(amr_enc.data.cpu().numpy())),("amr_enc \n",amr_enc)
        src_len , batch , src_rnn_size = src_enc.size()
        src_transformed = self.transform(src_enc.view(-1,src_rnn_size)).view(src_len,batch,-1).transpose(0,1)  #batch  x  src_len x  amr_rnn_size
        amr_enc = amr_enc.transpose(0,1).transpose(1,2) #batch    x amr_rnn_size x  amr_len
        score = src_transformed.bmm(amr_enc).transpose(1,2).transpose(0,1) #/ self.amr_rnn_size  #amr_len x batch  x  src_len
        assert not np.isnan(np.sum(score.data.cpu().numpy())),("score \n",score)
        final_score = gumbel_noise_sample(score)  if self.training else score
        assert not np.isnan(np.sum(final_score.data.cpu().numpy())),("final_score \n",final_score)
        if self.sink:
            posterior = sink_horn((final_score- (1-aligns)*1e6 ,lengths),k=self.sink,t=self.sink_t )
        else:
            final_score = final_score- (1-aligns)*1e6
            dim = final_score.size()
            final_score = final_score.view(-1, final_score.size(-1))
            posterior =self.sm(final_score).view(dim)
        return pack(posterior, lengths),pack(score,lengths) #amr_len x batch  x  src_len

#directly compute likelihood of concept being generated at words (a matrix for each training example)
def LikeliHood(tgtBatch,probBatch):
    '''tgtBatch:  data x  [n_feature + 1 (AMR_CAN_COPY)], batch_sizes
        probaBatch: (data x n_out, lengths ) *
            aligns:  amr_len x  batch x src_len , amr_l

        likelihood: data (amr) x src_len , batch_sizes
    '''

    batch_sizes = tgtBatch.batch_sizes
    likelihoods = []
    for i,prob in enumerate(probBatch):
        assert isinstance(prob, PackedSequence),"only support packed"
        if i == AMR_LE:
            prob_batch,lengths = unpack(prob)
            prob_batch = prob_batch.transpose(0,1)  #  batch x src_len x n_out
            n_out = prob_batch.size(-1)
            src_len = prob_batch.size(1)
            packed_index_data = tgtBatch.data[:,i].clamp(max=n_out-1) #so lemma not in high maps to last index ,data x 1

            copy_data = (packed_index_data<n_out-1).float()*tgtBatch.data[:,AMR_CAN_COPY].float()
            likes = []
            used = 0
            for batch_size in batch_sizes:  #n_different length in amr

                likelihood = prob_batch[:batch_size]   #batch_size x src_len x n_out
                batch_index = packed_index_data[used:used+batch_size].view(batch_size,1,1).expand(batch_size,src_len,1) #batch_size x src_len x 1
                like0 = likelihood.gather(2,batch_index).squeeze(2) #+  likelihood[:,:,-1]*high_and_copy    # will be corrected by posterior/align
                pointer = copy_data[used:used+batch_size].view(batch_size,1).expand(batch_size,src_len) # *alignBatch.data[used:used+batch_size] #batch_size x src_len x 1
                like =   likelihood[:,:,-1]*pointer + like0
                likes.append(like) #batch_size x src_len
                used = used + batch_size
            packed_likes = torch.cat(likes,0) #data x src_len

            likelihoods.append(packed_likes)
        else:
            prob_batch,lengths = unpack(prob)
            prob_batch = prob_batch.transpose(0,1)  #  batch x src_len x n_out
            src_len = prob_batch.size(1)
            packed_index_data = tgtBatch.data[:,i].contiguous()
            likes = []
            used = 0
       #     print (i,packed_index_data,batch_sizes)
            for batch_size in batch_sizes:  #n_different length in amr
                likelihood = prob_batch[:batch_size]   #batch_size x src_len x n_out
                batch_index = packed_index_data[used:used+batch_size].view(batch_size,1,1).expand(batch_size,src_len,1) #batch_size x src_len x 1
                likes.append(likelihood.gather(2,batch_index).squeeze(2)) #batch_size x src_len
                used = used + batch_size
            packed_likes = torch.cat(likes,0) #data x src_len
            likelihoods.append(packed_likes)
    likelihood = 1
    for i in range(0,len(likelihoods)):
        likelihood = likelihood  * likelihoods[i]
    return PackedSequence(likelihood,batch_sizes)

#compute variational posterior, and corresponding likelihood for concept identification, those are needed for computing loss
#also return the gumbel-sinkhorn input score matrix, which is needed for regularization
class VariationalAlignmentModel(nn.Module):
    #
    def __init__(self, opt,embs):
        super(VariationalAlignmentModel, self).__init__()
        self.txt_rnn_size = opt.txt_rnn_size
        self.jamr = opt.jamr
        if self.jamr :
            return
        self.posterior = Posterior( opt)
        self.amr_encoder = AmrEncoder( opt, embs)
        self.snt_encoder =  SentenceEncoder( opt, embs)


    def forward(self, srcBatch,tgtBatch,alignBatch,probBatch):
        assert isinstance(srcBatch,PackedSequence)

        likeli = LikeliHood(tgtBatch,probBatch)  # likelihood: data (amr) x src_len , batch_sizes
        if self.jamr :
            posterior,score = alignBatch,alignBatch
            posterior = PackedSequence(renormalize(posterior.data),posterior.batch_sizes)
            return posterior,likeli,score

        srcBatchData = srcBatch.data
        if srcBatchData.size(-1) == self.txt_rnn_size:
            src_enc = srcBatch
        else:
            src_enc = self.snt_encoder(srcBatch)
        amr_enc,hidden_t = self.amr_encoder(tgtBatch )
        posterior,score = self.posterior(src_enc,amr_enc,alignBatch)
        return posterior,likeli,score


#The entire model assembled
class AmrModel(nn.Module):

    def __init__(self,opt,embs):
        super(AmrModel, self).__init__()
        self.concept_decoder = ConceptIdentifier(opt, embs)
        self.poserior_m = VariationalAlignmentModel(opt, embs)
        self.opt = opt
        self.embs = embs
        self.rel = opt.rel
        if opt.rel :
            self.start_rel(opt)
        if opt.cuda:
            self.cuda()

    def start_rel(self,opt):
    #        return
        self.rel = True
        size = opt.txt_rnn_size
        assert int(size/2) == size/2
        embs = self.embs
        self.independent_posterior = opt.independent_posterior
        if opt.emb_independent:
            to_be_copied = ["lemma_lut","pos_lut","ner_lut","cat_lut"]
            for n in to_be_copied:
                embs[n] = deepcopy(self.embs[n])
                embs[n].requires_grad = True
            self.poserior_m.amr_encoder.cat_lut = embs["cat_lut"]
            self.poserior_m.amr_encoder.lemma_lut = embs["lemma_lut"]
            if not self.independent_posterior :
                self.poserior_m.snt_encoder  = deepcopy(self.concept_decoder.encoder)
            self.poserior_m.snt_encoder.pos_lut = embs["pos_lut"]
            self.poserior_m.snt_encoder.lemma_lut = embs["lemma_lut"]
            self.poserior_m.snt_encoder.ner_lut = embs["ner_lut"]
            #copy again for srl
            for n in to_be_copied:
                embs[n] = deepcopy(self.embs[n])
                embs[n].requires_grad = True
        self.relModel = RelModel(opt, embs )
        self.rel_encoder = RelSentenceEncoder(opt,embs)
        self.root_encoder = RootSentenceEncoder(opt,embs)

#srl index is from original node to recategorized, combine with posterior of recategorized concepts
    #we get the posterior needed for relation prediction
    def index_posterior(self,posterior,rel_index):
        '''rel_index:  # batch x var(amr_len)
            posterior :  pre_amr_len x  batch x src_len , pre_amr_l

            out:    batch x var(amr_len  x src_len ), amr_l
            '''
        posterior,lengths = unpack(posterior)

        out_l = [len(i) for i in rel_index]
        out_posterior = []
        for i,l in enumerate(out_l):
            out_posterior.append(posterior[:,i,:].index_select(0,rel_index[i]))


        return mypack(out_posterior,out_l)

    def forward(self, input,rel=False):

        if len(input)==3 and not rel:
            #training concept identification, alignBatch here indicating possibility of copying (or aligned by rule)
            srcBatch,tgtBatch,alginBatch = input
            probBatch,src_enc = self.concept_decoder(srcBatch)
            posteriors_likelihood_score = self.poserior_m(srcBatch,tgtBatch,alginBatch,probBatch )
            return probBatch,posteriors_likelihood_score,src_enc


        if len(input)==4 and rel:
            #training relation identification
            #  rel_index: concept_id -> re-categorized_id
            #  posterior: re-categorized_id -> alignment_soft_posterior
            rel_batch,rel_index,srcBatch,posterior = input
            assert not np.isnan(np.sum(posterior.data.data.cpu().numpy())),("posterior.data \n",posterior.data)
            posterior_data = renormalize(posterior.data+epsilon)
            assert not np.isnan(np.sum(posterior_data.data.cpu().numpy())),("posterior_data \n",posterior_data)
            posterior = PackedSequence(posterior_data,posterior.batch_sizes)
            indexed_posterior = self.index_posterior(posterior,rel_index)

            src_enc = self.rel_encoder(srcBatch,indexed_posterior)
            root_enc =  self.root_encoder(srcBatch)

            weighted_root_enc = self.root_posterior_enc(posterior,root_enc)
            weighted_enc= self.weight_posterior_enc(posterior,src_enc)   #src_enc MyDoublePackedSequence, amr_len

          #  self_rel_index = [  Variable(index.data.new(list(range(len(index))))) for       index in  rel_index]
            rel_prob = self.relModel(rel_batch,rel_index,weighted_enc,weighted_root_enc)
     #       assert not np.isnan(np.sum(rel_prob[0].data.data.cpu().numpy())),("inside srl\n",rel_prob[0].data.data)
            return rel_prob
        if len(input)==3 and rel:
            # relation identification evaluation
            rel_batch,srcBatch,alginBatch = input   #
            src_enc = self.rel_encoder(srcBatch,alginBatch)
            root_enc =  self.root_encoder(srcBatch)
            root_data,lengths = unpack(root_enc)
            mypacked_root_enc = mypack(root_data.transpose(0,1).contiguous(),lengths)
            rel_prob = self.relModel(rel_batch,alginBatch,src_enc,mypacked_root_enc)
            return rel_prob
        else:
        # concept identification evaluation
            srcBatch = input
            probBatch,src_enc= self.concept_decoder(srcBatch)
            return probBatch


    #encoding relaxation for root identification
    def root_posterior_enc(self,posterior,src_enc):
            '''src_enc:  # batch x var( src_l x dim)
                posterior =  pre_amr_len x  batch x src_len , amr_l

                out: batch x amr_len x txt_rnn_size
            '''
            posterior,lengths = unpack(posterior)
            enc,length_src = unpack(src_enc)
   #         print ("length_pairs",length_pairs)
    #        print ("lengths",lengths)
            weighted = []
            for i, src_l in enumerate(length_src): #src_len  x dim
                p = posterior[:,i,:src_l] #pre_full_amr_len  x src_len
                enc_t = enc[:src_l,i,:]
                weighted_enc = p.mm(enc_t)   #pre_amr_len  x dim
                weighted.append(weighted_enc)  #pre_amr_len   x dim
      #      print ("length_pairs",length_pairs)
            return mypack(weighted,lengths)

    #encoding relaxation for relation identification
    def weight_posterior_enc(self,posterior,src_enc):
            '''src_enc:  # batch x var(pre_amr_len x src_l x dim)
                posterior =  pre_amr_len x  batch x src_len , amr_l

                out: batch x amr_len x txt_rnn_size
            '''
            posterior,lengths = unpack(posterior)
            def handle_enc(enc):
                enc,length_pairs = doubleunpack(enc)
       #         print ("length_pairs",length_pairs)
        #        print ("lengths",lengths)
                dim = enc[0].size(-1)
                weighted = []
                new_length_pairs = []
                for i, src_enc_t in enumerate(enc):
                    p = posterior[:lengths[i],i,:] #pre_amr_len  x src_len
                    enc_trans = src_enc_t.transpose(0,1).contiguous().view(p.size(-1),-1) #src_len x (pre_amr_len  x dim)
                    weighted_enc = p.mm(enc_trans)   #pre_amr_len x (pre_amr_len  x dim)
                    weighted.append(weighted_enc.view(lengths[i],length_pairs[i][0],dim).transpose(0,1).contiguous())  #pre_amr_len x pre_amr_len  x dim
                    new_length_pairs.append([length_pairs[i][0],lengths[i]])
          #      print ("length_pairs",length_pairs)
                return doublepack(weighted,length_pairs)

            return handle_enc(src_enc)

