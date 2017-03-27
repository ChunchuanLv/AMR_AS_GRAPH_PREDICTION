import torch
import torch.nn as nn
from torch.autograd import Variable
from npmt.modules.Attention import *
from utility.constants import *

class Encoder(nn.Module):

    def __init__(self, opt, embs):
        self.layers = opt.enlayers
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
                        num_layers=opt.enlayers,
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


    def forward(self, input, hidden=None,posterior = None):
        batch_size = input.size(2) # batch first for multi-gpu compatibility
     #   print (input.size())
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
    def __init__(self,opt, embs,dicts):
        super(Generator, self).__init__()
        concept_ls = dicts["concept_ls"]
        self.rule_bias =  opt.rule_bias
        self.lemma_all =  opt.lemma_all
        self.cuda = opt.cuda
        self.dropout = nn.Dropout(opt.dropout)

        self.lemma_lut = embs["lemma_lut"]
        
        self.cat_lut = embs["cat_lut"]
        
        self.cat_non_linear  =  nn.Linear(self.cat_lut.embedding_dim,opt.rnn_size)
        self.cat_non_linear.weight.data.normal_(0,math.sqrt(2.0/(self.cat_lut.embedding_dim*opt.rnn_size)))
        self.lemma_linear = nn.Linear(self.lemma_lut.embedding_dim,opt.rnn_size)
        self.lemma_linear.weight.data.normal_(0,math.sqrt(2.0/(self.lemma_lut.embedding_dim*opt.rnn_size)))
        if self.rule_bias:
            self.rule_bias_w = nn.Linear(opt.rnn_size,1)
            self.rule_bias_w.weight.data.normal_(0,math.sqrt(2.0/(opt.rnn_size)))

        self.sig = nn.Sigmoid()

        self.softmax = nn.Softmax()
       
        if self.cuda:
            self.cat_non_linear = self.cat_non_linear.cuda()
            self.sig =  self.sig.cuda()
      #      self.rule_linear = self.rule_linear.cuda()
            self.lemma_linear = self.lemma_linear.cuda()
            self.softmax = self.softmax.cuda()
            if self.rule_bias:
                self.rule_bias_w =  self.rule_bias_w.cuda()
            
        self.concept_ids = Variable(torch.LongTensor(concept_ls).view(1,len(concept_ls)),requires_grad=False)
        self.n_freq = len(concept_ls)
        if self.cuda:
            self.concept_ids = self.concept_ids.cuda()
            
    def forward(self,attn,output,lemma):
        
        '''attn:   tgt_len x batch x src_len 
           output: tgt_len x batch x src_len x dim  

    tgt_split = torch.split(tgtBatch, opt.max_generator_batches,1)
            tgt: n_features, batch
            lemma: src_len x batch
            
            cat_prob: tgt_len x batch  x src_len x n_cat   keep src_len for posterior
        '''
        tgt_len = attn.size(0)
        batch_size = attn.size(1)
        src_len = attn.size(2)
        dim = output.size(3)
  #      print ("batch_size %.d  tgt_len %.d  src_len %.d dim %.d" % (batch_size,tgt_len,src_len,dim))
  #      lemma = lemma.t() #  batch x src_len
     #   print (lemma.size())
    #    print (self.cat_lut.weight.view(1,self.cat_lut.weight.size(0),self.cat_lut.weight.size(1)).size())
        cat_emb =self.cat_non_linear( self.cat_lut.weight) .t().contiguous() # dim x n_cat
        cat_likeli = self.softmax(output.view(-1,dim).mm(cat_emb))
        
        cat_likeli = cat_likeli.view(tgt_len,batch_size,src_len,cat_emb.size(1))

   #     del cat_emb
   #     print ("cat_prob",cat_prob.sum(3).squeeze(3).sum(2).squeeze(2))


        high_con_embed = self.lemma_linear(self.lemma_lut(self.concept_ids).view(-1 ,self.lemma_lut.embedding_dim)).t().contiguous()

 #       print ("output",output.view(tgt_len*batch_size*src_len,dim).size())
        high_con_score = output.view(-1,dim).mm(high_con_embed).view(-1,self.n_freq)#.view(tgt_len,batch_size,src_len,self.n_freq)



        source_lemma_embed= self.lemma_linear(self.lemma_lut(lemma).view(-1,self.lemma_lut.embedding_dim)).view(src_len,batch_size,dim)  #acting as target
     #   print ("source_lemma_embed",source_lemma_embed.size())   # 
        if self.lemma_all and  self.training:
            source_lemma_embed = source_lemma_embed.transpose(0,1).transpose(1,2).contiguous() #batch x dim  x src
            outputT = output.transpose(0,1).contiguous().view(-1,src_len*tgt_len,dim) #batch x ( tg x src) x dim
      #      print (source_lemma_embed.size(),outputT.size())
            rule_score = outputT.bmm(source_lemma_embed).view(batch_size,tgt_len,src_len,src_len).transpose(0,1).contiguous().view(-1,src_len)  #tg x batch x src x src
        else:
            outputT = output.transpose(0,2).contiguous().view(-1,tgt_len,dim)  #(src x batch  )x  tg x  x dim
            rule_score = outputT.bmm(source_lemma_embed.view(-1,dim,1)).view(src_len,batch_size,tgt_len,1).transpose(0,2).contiguous().view(-1,1) #( tg x batch x src ) x 1
                
        if  self.rule_bias: 
            rule_b = self.rule_bias_w(output.view(-1,dim))
            if (rule_score.size(1) != 1):
                rule_b = rule_b.expand(*rule_score.size())
            rule_score_b = rule_score + rule_b
            lemma_likeli =  self.softmax(torch.cat((high_con_score,rule_score_b),1))
        else:
            lemma_likeli = self.softmax(torch.cat((high_con_score,rule_score),1))
        
   #     print ("lemma_likeli",lemma_likeli.size())
        
        lemma_prob = torch.bmm(attn.view(-1,1,1),lemma_likeli.unsqueeze(1))
        
        lemma_prob = lemma_prob.view(tgt_len,batch_size,src_len,-1)

        out = torch.cat((lemma_prob,cat_likeli),3)
  #      del lemma_prob,lemma_likeli,total_score
        return out  #   tgt_len x batch x src_len x n_freq + 1 + n_cat
    
class Decoder(nn.Module):

    def __init__(self, opt, embs,concept_ls):
        self.layers = opt.delayers
        self.input_feed = opt.input_feed
        input_size = opt.lemma_dim + opt.cat_dim
        self.input_feed = False
        if self.input_feed:
            input_size += opt.rnn_size     #self feed posterior weighted encoder hidden
        self.srl = opt.srl
        if self.srl:
            input_size += opt.rel_dim
            self.rel_lut = embs["rel_lut"]
        super(Decoder, self).__init__()
        
        self.lemma_lut = embs["lemma_lut"]
        
        self.cat_lut = embs["cat_lut"]
        
        
        self.rnn = StackedLSTM(opt.delayers, input_size, opt.rnn_size , opt.dropout)
        self.attn = LocalAttention(opt.rnn_size )
        self.attn.setFlat(opt.flat)
        self.dropout = nn.Dropout(opt.dropout)
        self.Tensor = torch.FloatTensor

        self.cuda = opt.cuda
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                n = m.embedding_dim
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #    self.cat_nn = nn.Sequential( nn.Linear(opt.rnn_size,self.cat_lut.embedding_dim),
        #                            nn.Tanh())
    #    self.con_nn = nn.Sequential( nn.Linear(opt.rnn_size,self.cat_lut.embedding_dim),
        #                            nn.Tanh())
        if opt.cuda:
            self.Tensor =  torch.cuda.FloatTensor
            self.rnn.cuda()
            self.attn.cuda()
            self.dropout.cuda()
        #    self.cat_nn.cuda()
        #    self.con_nn.cuda()
        # self.rnn.bias_ih.data.div_(2)
        # self.rnn.bias_hh.data.copy_(self.rnn.bias_ih.data)

        self.hidden_size =  opt.rnn_size
    #
    #
    
    
    
    def forward(self, tgt, src ,hidden, context, init_output,mask):
        '''input:n_features,seq_len,batch
            emb:seq_len,batch,dim
            context: sourceL, batch, dim
            '''
        
        emb = torch.cat([self.lemma_lut(tgt[0]), self.cat_lut(tgt[1])], 2)
        batch_size = tgt.size(2)
        context = context.transpose(0,1).contiguous()  #batch x sourceL x dim

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output

        self.attn.applyMask(mask)
        attns = []
        for i, emb_t in enumerate(emb.chunk(emb.size(0), dim=0)):  #chunck make tensor list
            emb_t = emb_t.squeeze(0)         #emb_t: batch,dim
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)  #emb_t: batch,dim+(out_dim)  replacing output with posterior context

            output, hidden = self.rnn(emb_t, hidden)   #output:batch,dim
            outputContexted, attn = self.attn(output, context)   #outputContexted: batch , sourceL , [dim + context_dim]/2== context_dim
            outputContexted = self.dropout(outputContexted)
            if self.input_feed:
                sourceL = attn.size(1)
                dim = outputContexted.size(2)
                attn3 = attn.view(batch_size, 1,sourceL)  # batch x 1 x sourceL 
                weighted_Output = attn3.bmm(outputContexted).squeeze(1)  # batch  x dim
                output = weighted_Output
            attns += [attn]
            outputs += [outputContexted]
  #      del context,emb
        outputs = torch.stack(outputs)   
        attns = torch.stack(attns)
        return outputs, attns, hidden  #output: tgt_len x batch x src_len x dim   
                                       #attans   tgt_len x batch x src_len



class NMTModel(nn.Module):
    
    def initial_embedding(self):
        if self.initialize_lemma:
            import spacy
            nlp = spacy.load('en')     
        word_dict = self.dicts["word_dict"]
        lemma_dict = self.dicts["lemma_dict"]
        word_embedding = self.embs["word_fix_lut"]
        lemma_embedding = self.embs["lemma_lut"]
        word_initialized = 0
        lemma_initialized = 0
        with open(embed_path, 'r') as f:
            for line in f:
                parts = line.rstrip().split()
                id = word_dict[parts[0]]
                if id != UNK and id < word_embedding.num_embeddings:
                    tensor = torch.FloatTensor([float(s) for s in parts[1:]]).type_as(word_embedding.weight.data)
             #       print (parts,tensor.size())
                    word_embedding.weight.data[id].copy_(tensor)
                    word_initialized += 1
                if self.initialize_lemma:
                    id = lemma_dict[[i.lemma_ for i in nlp(parts[0])][0]]
                    if id != UNK and id < lemma_embedding.num_embeddings and self.initialize_lemma:
                        tensor = torch.FloatTensor([float(s) for s in parts[1:]]).type_as(lemma_embedding.weight.data)
                 #       print (parts,tensor.size())
                        lemma_embedding.weight.data[id].copy_(tensor)
                        lemma_initialized += 1
                    
                    
        print ("word_initialized", word_initialized)
        print ("lemma_initialized", lemma_initialized)
    def to_parallel(self,opt):
        self.encoder =  torch.nn.DataParallel(self.encoder, device_ids=opt.gpus)
        self.decoder =  torch.nn.DataParallel(self.decoder, device_ids=opt.gpus)
        self.generator =  torch.nn.DataParallel(self.generator, device_ids=opt.gpus)
    def __init__(self, opt,dicts):
        super(NMTModel, self).__init__()
        self.embs = dict()
        self.dicts = dicts
        self.initialize_lemma = False
        if  opt.initialize_lemma is not None:
            self.initialize_lemma =opt.initialize_lemma
        self.word_lut = nn.Embedding(dicts["word_dict"].size(),
                                  opt.word_dim,
                                  padding_idx=PAD)
        
        self.word_fix_lut = nn.Embedding(dicts["word_dict"].size(),
                                  opt.word_dim,
                                  padding_idx=PAD)
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
        self.rel_lut.weight.requires_grad = False      ##need load
        
        self.cat_lut = nn.Embedding(dicts["category_dict"].size(),
                                  opt.cat_dim,
                                  padding_idx=PAD)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                n = m.embedding_dim
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        self.generator = Generator( opt, self.embs,dicts)

    def set_generate(self, enabled,generator=None):
        self.generate = enabled
        if generator:
            self.generator = generator
        else:
            self.generator = Generator( self.opt, self.embs,self.dicts)

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
        tgt = input[1][:,:-1].contiguous()  #remove BOS
        mask = input[3]
   #     print (tgt.size())
        enc_hidden, context = self.encoder(src)
  #      print (enc_hidden[0].size(),enc_hidden[1].size(),context.size())
        init_output = self.make_init_decoder_output(context)
   #     print (init_output.size())
        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
    #    print (enc_hidden[0].size(),enc_hidden[1].size())

        out, attns,dec_hidden = self.decoder(tgt, src , enc_hidden, context, init_output,mask)
        if self.generate:
            out = self.generator(attns,out,tgt,src[LEMMA])
            return out
   #     del attns,dec_hidden
        return out,attns