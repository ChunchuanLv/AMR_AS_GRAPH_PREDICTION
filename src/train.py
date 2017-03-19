from npmt.Dict import *
import npmt
from npmt.data_iterator import *
from utility.constants import *
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time

parser = argparse.ArgumentParser(description='train.py')

## Data options

parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from',
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")

## Model options

parser.add_argument('-layers', type=int, default=1,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-srl', type=bool, default=False,
                    help='Wheather it is SRL stage')
parser.add_argument('-con_cat', type=bool, default=True,
                    help='Wheather conditional on cat to generate lemma')
parser.add_argument('-generate', type=bool, default=False,
                    help='Wheather it is generate stage')
parser.add_argument('-rnn_size', type=int, default=100,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_dim', type=int, default=100,
                    help='Word embedding sizes')
parser.add_argument('-lemma_dim', type=int, default=100,
                    help='Lemma embedding sizes')
parser.add_argument('-pos_dim', type=int, default=16,
                    help='Pos embedding sizes')
parser.add_argument('-ner_dim', type=int, default=16,
                    help='Ber embedding sizes')
parser.add_argument('-cat_dim', type=int, default=16,
                    help='Pos embedding sizes')
parser.add_argument('-rel_dim', type=int, default=16,
                    help='Ber embedding sizes')
parser.add_argument('-input_feed', type=int, default=0,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
# parser.add_argument('-residual',   action="store_true", , type=bool, default=True
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', type=bool, default=True,
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")

## Optimization options

parser.add_argument('-total_size', type=int, default=1024,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=16,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-learning_rate', type=float, default=1,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.1""")
parser.add_argument('-max_grad_norm', type=float, default=1,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.5,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-learning_rate_decay', type=float, default=0.998,
                    help="""Decay learning rate by this much if (i) perplexity
                    does not decrease on the validation set or (ii) epoch has
                    gone past the start_decay_at_limit""")
parser.add_argument('-start_decay_at', default=8,
                    help="Start decay after this epoch")
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")

# GPU
parser.add_argument('-gpus',  nargs='*',default=[1], type=int,
                    help="Use CUDA")

parser.add_argument('-log_interval', type=int, default=10,
                    help="Print stats at this interval.")
# parser.add_argument('-seed', type=int, default=3435,
#                     help="Seed for random initialization")

opt = parser.parse_args()

opt.cuda = len(opt.gpus)

print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with -cuda")

if opt.cuda:
    cuda.set_device(opt.gpus[0])

def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.cuda:
        crit.cuda()
    return crit

def my_loss(out,high_index,rule_index,lemma_batch,cat_batch,n_freq,n_cat,eval):
    ''' high_prob:  tgt_len x batch x src_len x n_freq  ,
        rule_prob:  tgt_len x batch x src_len,  
        cat_prob:   tgt_len x  batch x src_len x n_cat
        high_index: tgt_len x batch_size        (0 ... n_high-1)
        rule_index: tgt_len x batch_size x src_len  (0 1)
        cat_batch:  tgt_len x batch_size
        lemma_batch:  tgt_len x batch_size
        posterior:  tgt_len x batch x src_len    
    '''

    tgt_len = out.size(0)
    batch_size =  out.size(1)
    total_size = tgt_len*batch_size
    src_len = out.size(2)
 #   print ("tgtBatch",tgtBatch.size())
    lemma_batch = lemma_batch.view(-1)
    cat_batch = cat_batch.view(-1)

    high_index = high_index.view(-1)
    rule_index = rule_index.view(-1,src_len)

    high_prob = out[:,:,:,0:n_freq].contiguous().view(-1,src_len,n_freq)
    rule_prob = out[:,:,:,n_freq].contiguous().view(-1,src_len)
    cat_likeli = out[:,:,:,n_freq+1:].contiguous().view(-1,src_len,n_cat)

  #  high_prob = Variable(high_prob.data, requires_grad= not eval , volatile= eval)
   # rule_prob = Variable(rule_prob.data, requires_grad=  not eval , volatile= eval)
  #  cat_prob = Variable(cat_prob.data, requires_grad=  not eval , volatile= eval)

 #   print (rule_prob.size(),rule_index.size())
    effective_rule = rule_prob*rule_index.float() 
    
    effective_high = high_prob.gather(2,high_index.view(total_size,1,1).expand(total_size,src_len,1)).squeeze(2)
    
    effective_lemma = effective_rule+effective_high +1e-8
#    print("effective_lemma",effective_lemma.min(),effective_lemma.max(),effective_lemma.sum(2).squeeze(2))
  #  print ("cat_prob",cat_prob.size())
   # print ("cat_batch",cat_batch.size())
    effective_cat = cat_likeli.gather(2,cat_batch.view(total_size,1,1).expand(total_size,src_len,1)).squeeze(2)


    effective = effective_lemma*effective_cat   #prior term is computed in effective_lemma
  #  print (effective.min(0))
  #  print (lemma_batch!=0)
  #  print ("n_cat,n_freq",n_cat,n_freq)

    num_words = lemma_batch.data.ne(PAD).sum()
    loss = -(effective.sum(1).squeeze(1)[lemma_batch!=PAD]).log().sum()
 #   print ("loss",loss)
   # print ("loss",-effective.sum(2).squeeze(2).log().sum()/tgt_len/batch,loss)
 #   print (effective.size())
    posterior = effective/effective.sum(1).expand(total_size,src_len)
    posterior = posterior.view(tgt_len,batch_size,src_len)
    if not eval and False:
        loss.backward()
        grad = torch.cat((high_prob.grad.data,rule_prob.grad.data.unsqueeze(2),cat_likeli.grad.data),2).view(tgt_len,batch_size,src_len,-1)
    #    del high_prob,rule_prob,cat_prob,effective,effective_cat,effective_lemma
        return loss,num_words,posterior,grad

 #   del high_prob,rule_prob,cat_prob,effective,effective_cat,effective_lemma
    return loss,num_words,posterior

def memoryEfficientLoss(outputs,attns,tgtBatch, srcBatch, high_index,lemma_index,generator,dicts,eval):
    ''' high_index: tgt_len x batch_size        (0 ... n_high-1)
        rule_index: tgt_len x batch_size x src_len  (0 1)
        cat_batch:  tgt_len x batch_size
        lemma_batch:  tgt_len x batch_size
        posterior:  tgt_len x batch x src_len
    '''
    loss,num_words = 0,0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval).contiguous()
    batch_size = outputs.size(1)




    outputs_split = torch.split(outputs, opt.max_generator_batches,0)
    attns_split =  torch.split(attns, opt.max_generator_batches,0)
    lemma_batch = tgtBatch[0]
    cat_batch = tgtBatch[1]
    lemma_split = torch.split(lemma_batch, opt.max_generator_batches,0)
    cat_split = torch.split(cat_batch, opt.max_generator_batches,0)
    high_index_split = torch.split(high_index, opt.max_generator_batches,0)
    lemma_index_split = torch.split(lemma_index,opt.max_generator_batches,0)

    out_grads = []
    posteriors = []
    for out_t, lemma_t,cat_t ,attn_t,high_index_t,lemma_index_t in zip(outputs_split, lemma_split,cat_split,attns_split,high_index_split,lemma_index_split):

        out = generator(attn_t,out_t,srcBatch[LEMMA])
  #      print ([i.size() for i in [out,out_t, tgt_t ,attn_t,high_index_t,lemma_index_t]])
        loss_t,num_words_t,posterior = my_loss(out,high_index_t,lemma_index_t,lemma_t,cat_t,len(dicts["concept_ls"]),len(dicts["category_dict"]),eval)

        loss += loss_t.data[0]
        num_words += num_words_t
        posteriors += [posterior]
        if not eval:
            out_grads += [loss_t.backward(retain_variables=True)]

    grad_output = None if outputs.grad is None else outputs.grad.data

    posteriors = torch.cat(posteriors)

    if eval:
        return loss,num_words,posteriors
    return loss,num_words,posteriors,grad_output

def eval(model, data,dicts):
    total_loss = 0
    total_words = 0

    model.eval()
    for i in range(len(data)):
        
        x = data[i]
        srcBatch, tgtBatch, idBatch = x
        high_index,lemma_index = idBatch
        tgtBatch = tgtBatch[:,1:]
        high_index = high_index[1:]
        lemma_index = lemma_index[1:]

        out,attns= model.forward(x)

        generator = model.generator
   #     print ("out",out.size())

   #     print ("in eval tgtBatch",tgtBatch.size())
        loss,num_words,posterior  = memoryEfficientLoss(out,attns,tgtBatch, srcBatch, high_index,lemma_index,generator,dicts,True)

        total_loss += loss
        total_words += num_words

    model.train()
    return total_loss / total_words

def trainModel(model, trainData, validData, dicts, optim,valid_loss_low =math.exp(100)):
    print(model)
    model.train()
    start_time = time.time()
    def trainEpoch(epoch):

        # shuffle mini batch order
   #     print (len(trainData))
        batchOrder = torch.randperm(len(trainData))

        total_loss, report_loss = 0, 0
        total_words, report_words = 0, 0
        start = time.time()
        for i in range(len(trainData)):

            batchIdx = batchOrder[i] if epoch >= opt.curriculum else i

            model.zero_grad()
            
            x = trainData[batchIdx]
            srcBatch, tgtBatch, idBatch = x
            high_index,lemma_index = idBatch
            tgtBatch = tgtBatch[:,1:]
            high_index = high_index[1:]
            lemma_index = lemma_index[1:]
     #       print ("trainModel",srcBatch.size())
            generator = model.generator
            out,attns = model.forward(x)
            loss,num_words,posterior,out_grad  = memoryEfficientLoss(out,attns,tgtBatch, srcBatch, high_index,lemma_index,generator,dicts,False)
       #     print ("in trainModel tgtBatch",tgtBatch.size())
         #   print ("out",out.size())

       #     print ("out",out.size())

       #     print ("in eval tgtBatch",tgtBatch.size())
        #    loss,num_words,posterior,out_grad = my_loss(out,high_index,lemma_index,tgtBatch,len(dicts["concept_ls"]),len(dicts["category_dict"]),False)
        #    print (grad)
       #     print ("out_grad",out_grad.size())
            out.backward(out_grad)
            # update the parameters
            grad_norm = optim.step()
            report_loss += loss
            total_loss += loss
            total_words += num_words
            report_words += num_words
     #       del out,out_grad,x
            if i % opt.log_interval == 0 and i > 0 :
        #        print ("trainModel",srcBatch.size())
                print("Epoch " + str(epoch)+" "+ str(i) +"//"+ str(len(trainData)))
                print("perplexity", math.exp(1.0*report_loss/report_words))
                print("tokens/s", 1.0*report_words/(time.time()-start))
                print(str(time.time()-start_time) +" s elapsed")
                
                report_loss = report_words = 0
                start = time.time()

        return total_loss / total_words
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        #  (1) train for one epoch on the training set
        train_loss = trainEpoch(epoch)
        print('Train perplexity: %g' % math.exp(train_loss))

        #  (2) evaluate on the validation set
        
        valid_loss = eval(model, validData,dicts)
        valid_ppl = math.exp(valid_loss)
        print('Validation perplexity: %g' % valid_ppl)

        #  (3) maybe update the learning rate
        if opt.optim == 'sgd':
            optim.updateLearningRate(valid_loss, epoch)
        if (valid_loss<valid_loss_low):
            #  (4) drop a checkpoint
            checkpoint = {
                'model': model,
                'opt': opt,
                'epoch': epoch,
                'optim': optim,
            }
            print ("saving")
            print('Validation perplexity: %g' % valid_ppl)
            print("Epoch: ", epoch)
            torch.save(checkpoint,
                       'model/valid_best.pt' )
            valid_loss_low = valid_loss

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
    
    training_data = DataIterator(trainingFilesPath,total_size = opt.total_size,cuda = opt.cuda,volatile = False,dicts=dicts)
    
    dev_data = DataIterator(devFilesPath,total_size = opt.total_size,cuda = opt.cuda,volatile = True,dicts=dicts)
    
    print("word_dict %d lemma_dict %d pos_dict %d ner_dict %d concept_dict %d "% (len(word_dict),len( lemma_dict),len( pos_dict),len( ner_dict ),len(concept_dict)))
    print(' * number of training sentences. %d' %
          len(training_data.src))
    print(' * maximum total size. %d' % opt.total_size)
    print('Building model...')
    valid_loss = math.exp(100)
    if opt.train_from is None:
        NMTModel = npmt.Models.NMTModel(opt, dicts)

        NMTModel.initial_embedding()

  #      NMTModel.to_parallel(opt)
        if opt.cuda:
            NMTModel.cuda()
        else:
            NMTModel.cpu()
            
    #    for p in NMTModel.parameters():
      #      p.data.uniform_(-opt.param_init, opt.param_init)
            
        parameters_to_train = []
        for p in NMTModel.parameters():
            if p.requires_grad:
                parameters_to_train.append(p)
    #    print ([p.size() for p in parameters_to_train])
        optim = npmt.Optim(
            parameters_to_train, opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
        
 #       print ("posterior[:,0,:]",posterior[:,0,:])
    else:
        print('Loading from checkpoint at %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from)
        NMTModel = checkpoint['model']
        if opt.cuda:
            NMTModel.cuda()
        else:
            NMTModel.cpu()
        optim = checkpoint['optim']
        opt.start_epoch = checkpoint['epoch'] + 1
        valid_loss = eval(NMTModel, dev_data,dicts)
        valid_ppl = math.exp(valid_loss)
        print('Validation perplexity: %g' % valid_ppl)
        
    nParams = sum([p.nelement() for p in NMTModel.parameters()])
    print('* number of parameters: %d' % nParams)

    trainModel(NMTModel, training_data,  dev_data, NMTModel.dicts, optim,valid_loss)


if __name__ == "__main__":
    main()