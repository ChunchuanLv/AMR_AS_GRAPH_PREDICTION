import torch
import torch.nn as nn
from torch.autograd import Variable
from npmt.modules.Attention import *
from utility.constants import *

class Encoder(nn.Module):

    def __init__(self, opt, embs):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        print ("self.num_directions ",self.num_directions )
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        inputSize = opt.word_dim*2 + opt.lemma_dim + opt.pos_dim +opt.ner_dim 
        self.srl = opt.srl
        if opt.srl == True:
            inputSize += 1
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(inputSize, self.hidden_size,
                        num_layers=opt.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)
        
        
        self.word_lut = embs["word_lut"]
        
        self.word_fix_lut  = embs["word_fix_lut"]
        
        self.lemma_lut = embs["lemma_lut"]
        
        self.pos_lut = embs["pos_lut"]
        
        self.ner_lut = embs["ner_lut"]
        
        if opt.cuda:
            self.rnn.cuda()
        # self.rnn.bias_ih_l0.data.div_(2)
        # self.rnn.bias_hh_l0.data.copy_(self.rnn.bias_ih_l0.data)

        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.copy_(pretrained)

    def forward(self, input, hidden=None,posterior = None):
        batch_size = input.size(2) # batch first for multi-gpu compatibility
        print (input.size())
        word_embed = self.word_lut(input[WORD])
        word_fix_embed = self.word_fix_lut(input[WORD])
        lemma_emb = self.lemma_lut(input[LEMMA])
        pos_emb = self.pos_lut(input[POS])
        ner_emb = self.ner_lut(input[NER])
        if self.srl:
            emb = torch.cat([word_embed,word_fix_embed,lemma_emb,pos_emb,ner_emb,posterior],2)
        else:
            emb = torch.cat([word_embed,word_fix_embed,lemma_emb,pos_emb,ner_emb],2)
        if hidden is None:
            h_size = (self.layers * self.num_directions, batch_size, self.hidden_size)
            h_0 = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)
            c_0 = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)
            hidden = (h_0, c_0)

        outputs, hidden_t = self.rnn(emb, hidden)
        return hidden_t, outputs


            
class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.input_size =input_size
        for i in range(num_layers):
            layer = nn.LSTMCell(input_size, rnn_size)
            self.add_module('layer_%d' % i, layer)
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i in range(self.num_layers):
            layer = getattr(self, 'layer_%d' % i)
    #        print (i,layer.input_size,layer.hidden_size,input.size(),h_0[i].size(),c_0[i].size())
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)

class Generator(nn.Module):
    def __init__(self,opt, embs):
        super(Generator, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        self.lemma_lut = embs["lemma_lut"]
        
        self.cat_lut = embs["cat_lut"]
        
        self.cat_linear = nn.Linear(opt.rnn_size, self.cat_lut.embedding_dim)
        
        self.lemma_linear = nn.Linear(opt.rnn_size, self.lemma_lut.embedding_dim)
                                
    #input: batch x sourceL x dim
    #index: batch x []
    def forward(self, input, index):
        input = input.view(input.size(0)*input.size(1),input.size(2))
        cat_emb = self.cat_linear(input)
        lemma_emb = self.lemma_linear(input)
        for i in range(self.num_layers):
            layer = getattr(self, 'layer_%d' % i)
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Decoder(nn.Module):

    def __init__(self, opt, embs,concept_ls):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.lemma_dim + opt.cat_dim
        if self.input_feed:
            input_size += opt.rnn_size     #self feed posterior weighted encoder hidden
        self.srl = opt.srl
        if self.srl:
            input_size += opt.rel_dim
            self.rel_lut = embs["rel_lut"]
        super(Decoder, self).__init__()
        
        self.lemma_lut = embs["lemma_lut"]
        
        self.cat_lut = embs["cat_lut"]
        
        self.concept_ids = Variable(torch.LongTensor(concept_ls).view(1,len(concept_ls)))
        
        self.concept_lut = torch.LongTensor(self.lemma_lut.num_embeddings).fill_(len(concept_ls))
        for i, id_con in enumerate(concept_ls):  #chunck make tensor list
            self.concept_lut[id_con] = i
        self.cat_non_linear =  nn.Sequential(nn.Linear(self.cat_lut.embedding_dim,opt.rnn_size),
                                         nn.ReLU())
        
        self.lemma_non_linear =nn.Sequential( nn.Linear(self.lemma_lut.embedding_dim,opt.rnn_size),
                                              nn.ReLU())
                                
        
        self.rnn = StackedLSTM(opt.layers, input_size, opt.rnn_size , opt.dropout)
        self.attn = LocalAttention(opt.rnn_size )
        self.dropout = nn.Dropout(opt.dropout)
        self.nhigh = self.concept_ids.size(1)
        self.Tensor = torch.FloatTensor
        if opt.cuda:
            print ("to cuda")
            self.Tensor =  torch.cuda.FloatTensor
            self.concept_lut = self.concept_lut.cuda()
            self.concept_ids = self.concept_ids.cuda()
            self.cat_non_linear.cuda()
            self.lemma_non_linear.cuda()
            self.rnn.cuda()
            self.attn.cuda()
            self.dropout.cuda()
        # self.rnn.bias_ih.data.div_(2)
        # self.rnn.bias_hh.data.copy_(self.rnn.bias_ih.data)

        self.hidden_size =  opt.rnn_size
    #
    #
    
    
    def get_posterior(self,attn,index,outputContexted,tgt,lemma):
        '''attn: batch x sourceL 
            outputContexted: batch x sourceL x dim
            index:   batch, []
            tgt: n_features, batch
            lemma: sourceL x batch
            
            high_index :  batch_size
            lemma_index : batch_size x sourceL
        '''
        high_index,lemma_index = index
        batch_size = attn.size(0)
        sourceL = attn.size(1)
        dim = outputContexted.size(2)
    #    print (self.cat_lut.weight.view(1,self.cat_lut.weight.size(0),self.cat_lut.weight.size(1)).size())
        cat_emb =self.cat_non_linear( self.cat_lut.weight) .t().contiguous() # dim x type
        print ("cat_emb",cat_emb.size())
        
        cat_index = tgt[1]
        cat_exp = outputContexted.view(batch_size*sourceL,dim).mm(cat_emb).exp_().view(batch_size,sourceL,cat_emb.size(1))
        
        cat_actual_exp = cat_exp.gather(2,cat_index.view(batch_size,1,1).expand(batch_size,sourceL,1)).squeeze(2)
        print (cat_exp.size(),cat_actual_exp.size())
        un_summed_cat_likelihood = cat_actual_exp/cat_exp.sum(2).squeeze(2)*attn
        cat_likelihood = un_summed_cat_likelihood.sum(1)
        
        print (cat_likelihood)
        
        high_con_embed = self.lemma_non_linear(self.lemma_lut(self.concept_ids).view(self.concept_ids.size(1)\
                                                                                     ,self.lemma_lut.embedding_dim))
        
        source_lemma_embed= self.lemma_non_linear(self.lemma_lut(lemma.t()).view(batch_size*sourceL,\
                                                                                  self.lemma_lut.embedding_dim))  #   (batch x sourceL) x dim
        
        rule_exp = torch.sum(outputContexted.view(batch_size*sourceL,dim)*source_lemma_embed,1).exp_().view(batch_size,sourceL)
        
   #     print (cat_emb.size(),high_con_embed.size(),outputContexted.size(),tgt.size())
        high_exp = outputContexted.view(batch_size*sourceL,dim).mm(high_con_embed.t().contiguous()).exp_().view(batch_size,sourceL,self.nhigh)
        zero_tensor = Variable(self.Tensor(batch_size,sourceL,1).zero_(),requires_grad=False)
        
        high_total_exp = torch.cat((high_exp,zero_tensor),2).view(batch_size,sourceL,self.nhigh+1)
        
        high_actual_exp = high_total_exp.gather(2,high_index.view(batch_size,1,1).expand(batch_size,sourceL,1)).squeeze(2)
        
        rule_actual_exp = lemma_index.float()*rule_exp
        
        total_actual_exp = high_actual_exp+rule_actual_exp 
        
        total_exp = high_total_exp.sum(2).squeeze(2) +rule_exp
        
        un_summed_likelihood = total_actual_exp/total_exp*attn
        likelihood = un_summed_likelihood.sum(1)
        posterior = un_summed_likelihood/likelihood.expand(batch_size,sourceL)
#        print ("lemma_index,rule_actual_exp,rule_total_exp",lemma_index,rule_actual_exp,rule_total_exp)
        loglikelihood = torch.log(likelihood).squeeze(1)
     #   print ("posterior,logsoftmax",posterior,loglikelihood)
      #  lemma_total_exp.add_()
        return posterior, loglikelihood,cat_likelihood  #batch x sourceL  , batch
    
    def forward(self, input,src, index,hidden, context, init_output):
        '''input:n_features,seq_len,batch
            index:  seq_len, batch_size, []
            emb:seq_len,batch,dim
            context: sourceL, batch, dim
            '''
        
        emb = torch.cat([self.lemma_lut(input[0]), self.cat_lut(input[1])], 2)
        batch_size = input.size(1)
        context = context.transpose(0,1)  #batch x sourceL x dim
        h_size = (batch_size, self.hidden_size)
        output = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output
        posteriors = []
        logsoftmaxes = []
        for i, emb_t in enumerate(emb.chunk(emb.size(0), dim=0)):  #chunck make tensor list
            if i >= emb.size(0) - 2:
                break
            emb_t = emb_t.squeeze(0)         #emb_t: batch,dim
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)  #emb_t: batch,dim+(out_dim)  replacing output with posterior context

            output, hidden = self.rnn(emb_t, hidden)   #output:batch,dim 
            outputContexted, attn = self.attn(output, context)   #output: batch , sourceL , [dim + context_dim]/2== context_dim
            outputContexted = self.dropout(outputContexted)
            posterior,lemma_likelihood,cat_likelihood = self.get_posterior(attn,(index[0][i],index[1][i]),outputContexted,tgt=input[:,i,:],lemma = src[LEMMA]) # exclude last target from inputs)
            
            batch_size = attn.size(0)
            sourceL = attn.size(1)
            dim = outputContexted.size(2)
            posterior3 = posterior.view(batch_size, 1,sourceL)  # batch x 1 x sourceL 
            weighted_Output = posterior3.bmm(outputContexted).squeeze(1)  # batch  x dim
            output = weighted_Output
            posteriors += [posterior]
            outputs += [output]
            logsoftmaxes += [lemma_likelihood]

        outputs = torch.stack(outputs)   #output: seq_len,batch,dim
        logsoftmaxes = torch.stack(logsoftmaxes)
        return outputs.transpose(0, 1), hidden, posteriors,logsoftmaxes  #output: batch,seq_len,dim   want batch,src_len,seq_len,dim



def initial_embedding(word_embedding,word_dict):
    with open(embed_path, 'r') as f:
        for line in f:
            parts = line.rstrip().split()
            id = word_dict[parts[0]]
            if id and id < word_embedding.num_embeddings:
                word_embedding.weight.data[id].copy_(torch.FloatTensor([float(s) for s in parts[1:]]).type_as(word_embedding.weight.data))
            
class NMTModel(nn.Module):

    def __init__(self, opt,dicts):
        super(NMTModel, self).__init__()
        self.embs = dict()
        
        self.word_lut = nn.Embedding(dicts["word_dict"].size(),
                                  opt.word_dim,
                                  padding_idx=PAD)
        
        self.word_fix_lut = nn.Embedding(dicts["word_dict"].size(),
                                  opt.word_dim,
                                  padding_idx=PAD)
        initial_embedding(self.word_fix_lut,dicts["word_dict"])
        self.word_fix_lut.weight.requires_grad = False      ##need load
        
        
        self.lemma_lut = nn.Embedding(dicts["lemma_dict"].size(),
                                  opt.lemma_dim,
                                  padding_idx=PAD)
        
        self.pos_lut = nn.Embedding(dicts["pos_dict"].size(),
                                  opt.pos_dim,
                                  padding_idx=PAD)
        
        self.ner_lut = nn.Embedding(dicts["ner_dict"].size(),
                                  opt.ner_dim,
                                  padding_idx=PAD)
        
        self.rel_lut = nn.Embedding(dicts["rel_dict"].size(),
                                  opt.rel_dim,
                                  padding_idx=PAD)
        
        self.cat_lut = nn.Embedding(dicts["category_dict"].size(),
                                  opt.cat_dim,
                                  padding_idx=PAD)
        
        if opt.cuda:
            self.word_lut.cuda()
            self.word_fix_lut.cuda()
            self.lemma_lut.cuda()
            self.pos_lut.cuda()
            self.ner_lut.cuda()
            self.rel_lut.cuda()
            self.cat_lut.cuda()
        self.embs["word_lut"] = self.word_lut
        self.embs["word_fix_lut"] = self.word_fix_lut
        self.embs["lemma_lut"] = self.lemma_lut
        self.embs["pos_lut"] = self.pos_lut
        self.embs["ner_lut"] = self.ner_lut
        self.embs["rel_dict"] = self.rel_lut
        self.embs["cat_lut"] = self.cat_lut
        
            
        self.encoder = Encoder( opt, self.embs)
        self.decoder = Decoder( opt, self.embs,dicts["concept_ls"])
        self.generate = opt.generate

    def set_generate(self, enabled,generator=None):
        self.generate = enabled
        if generator:
            self.generator = generator
        else:
            self.generator = Generator( opt, self.embs)

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, input):
        src = input[0]
        tgt = input[1]
   #     print (tgt.size())
        index = input[2]
        enc_hidden, context = self.encoder(src)
        print (enc_hidden[0].size(),enc_hidden[1].size(),context.size())
        init_output = self.make_init_decoder_output(context)
   #     print (init_output.size())

        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
        print (enc_hidden[0].size(),enc_hidden[1].size())

        out, dec_hidden, attns,logsoftmaxes = self.decoder(tgt,src,index, enc_hidden, context, init_output)
        if self.generate:
            out = logsoftmaxes

        return out