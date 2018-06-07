#!/usr/bin/env python3.6
# coding=utf-8
'''

Scripts to train the model

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

from parser.Dict import *

from parser.DataIterator import *
import parser
import argparse
from src import *
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
from torch import cuda
from torch.autograd import Variable
import time
from utility.Naive_Scores import *

def train_parser():
    parser = argparse.ArgumentParser(description='train')
    ## Data options
    parser.add_argument('-suffix', default=".txt_pre_processed", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('-folder', default=allFolderPath, type=str ,
                        help="""the folder""")
    parser.add_argument('-jamr', default=0, type=int,
                        help="""wheather to use fixed alignment""")
    parser.add_argument('-save_to', default=save_to,
                        help="""folder to save""" )
    parser.add_argument('-train_from', default = train_from,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model.""")
    parser.add_argument('-get_wiki', type=bool,default=True)
    parser.add_argument('-get_sense', type=bool,default=True)

    ## Model optionsinitialize_lemma






    parser.add_argument('-cat_bias', type=int, default=1,
                        help='Wheather bias category')

    parser.add_argument('-lemma_bias', type=int, default=0,
                        help='Wheather bias lemma')



    parser.add_argument('-independent_posterior', type=int, default=1,
                        help='Wheather normalize over all lemmas')

    parser.add_argument('-train_posterior', type=int, default=1,
                        help='keep training posterior')

    parser.add_argument('-alpha', type=float, default=0.1,
                        help='unk with alpha ')

    parser.add_argument('-initialize_word', type=int, default=1,
                        help='Wheather initialize_lemma')

    #layers
    parser.add_argument('-rel_enlayers', type=int, default=2,
                        help='Number of layers in the rel LSTM encoder/decoder')
    parser.add_argument('-root_enlayers', type=int, default=1,
                        help='Number of layers in the root LSTM encoder/decoder')
    parser.add_argument('-txt_enlayers', type=int, default=1,
                        help='Number of layers in the concept LSTM encoder/decoder')
    parser.add_argument('-amr_enlayers', type=int, default=1,
                        help='Number of layers in the amr LSTM encoder/decoder')

    parser.add_argument('-txt_rnn_size', type=int, default=512)
    parser.add_argument('-rel_rnn_size', type=int, default=512,
                        help='Number of hidden units in the rel/root LSTM encoder/decoder')
    parser.add_argument('-amr_rnn_size', type=int, default=200)

    parser.add_argument('-rel', type=float, default=1.0,
                        help='Wheather train relation/root identification')

    #dimensions
    parser.add_argument('-word_dim', type=int, default=300,
                        help='Word embedding sizes')
    parser.add_argument('-dim', type=int, default=200,
                        help='lemma/high embedding sizes')
    parser.add_argument('-pos_dim', type=int, default=32,
                        help='Pos embedding sizes')
    parser.add_argument('-ner_dim', type=int, default=16,
                        help='Ner embedding sizes')
    parser.add_argument('-cat_dim', type=int, default=32,
                        help='Pos embedding sizes')
    parser.add_argument('-rel_dim', type=int, default=200,
                        help='mixed amr node and text representation dimension')
    # parser.add_argument('-residual',   action="store_true", , type=bool, default=True
    #                     help="Add residual connections between RNN layers.")
    parser.add_argument('-brnn', type=bool, default=True,
                        help='Use a bidirectional encoder')

    ## Optimization options

    parser.add_argument('-weight_decay', type=float, default=0.00001,
                        help='l2 weight_decay')

    parser.add_argument('-train_all', type=bool, default=False,
                        help='wheather to train all parameters. useful when reloading model for train')

    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('-epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')

    parser.add_argument('-optim', default='adam',
                        help="Optimization method. [sgd|adagrad|adadelta|adam]")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Starting learning rate. If adagrad/adadelta/adam is
                        used, then this is the global learning rate. Recommended
                        settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.1""")
    parser.add_argument('-max_grad_norm', type=float, default=10,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('-dropout', type=float, default=0.2,
                        help='Dropout probability; applied between LSTM stacks and some MLPs.')

    parser.add_argument('-sink', type=int, default=10,
                        help='steps of sinkhorn procedure')

    parser.add_argument('-sink_t', type=float, default=1,
                        help='gumbel-sinkhorn temperature')

    parser.add_argument('-prior_t', type=float, default=5,
                        help='prior tempeture for gumbel-sinkhorn')

    parser.add_argument('-sink_re', type=float, default=10,
                        help='gumbel-sinkhorn finite step regularzor penalizing non double-stochaticity')

    parser.add_argument('-learning_rate_decay', type=float, default=0.98,
                        help="""Decay learning rate by this much if (i) perplexity
                        does not decrease on the validation set or (ii) epoch has
                        gone past the start_decay_at_limit""")

    parser.add_argument('-start_decay_at', default=5,
                        help="Start decay after this epoch")

    parser.add_argument('-emb_independent', type=int, default=1,
                        help="""wheather relation system use independent embedding""")

    # GPU
    parser.add_argument('-gpus',  nargs='*',default=[0], type=int,
                        help="Use ith gpu, if -1 then load in cpu")  #training probably cannot work on cpu
    parser.add_argument('-from_gpus',  nargs='*',default=[0], type=int,
                        help="model load from which gpu, must be specified when current gpu id is different from saving time")

    parser.add_argument('-log_per_epoch', type=int, default=10,
                        help="Print stats at this interval.")

    parser.add_argument('-renyi_alpha', type=float,default=.5,
                        help="parameter of renyi_alpha relaxation, "
                             "which is alternative to hierachical relaxation in the paper")


    return parser

def posterior_regularizor(posterior):
    '''probBatch:   tuple (src_len x  batch x n_out,lengths),
       tgtBatch: amr_len x batch x n_feature  , lengths
        posterior = packed( amr_len x  batch x src_len , lengths)

      total_loss,total_data
    '''

    assert isinstance(posterior, tuple),"only support tuple"
    unpacked_posterior,lengths = unpack(posterior)
    activation =pack(unpacked_posterior.sum(0),lengths,batch_first=True)[0]
    activation_loss = torch.nn.functional.relu(activation-1).sum()
    return activation_loss

import math


def sinkhorn_score_regularizor(score):
    '''probBatch:   tuple (src_len x  batch x n_out,lengths),
       tgtBatch: amr_len x batch x n_feature  , lengths
        score = packed( amr_len x  batch x src_len , lengths)

      total_loss,total_data
    '''

    scores,lengths = unpack(score)
    S = 0
    r= opt.prior_t/opt.sink_t
    gamma_r = math.gamma(1+r)
    for i,l in enumerate(lengths):
        S = S + r / scores[:l,i,:l].sum()+gamma_r*torch.exp( -scores[:l,i,:l]*r).sum()
    return S #+activation_loss

epsilon = 1e-6
import numpy as np
def Total_Loss(probs,tgtBatch,alginBatch,epoch,rel_roles_batch = None,roots_log_soft_max=None,gold_roots=None):
    '''probBatch:   tuple (src_len x  batch x n_out,lengths),
       tgtBatch: amr_len x batch x n_feature  , lengths
        posterior =  amr_len x  batch x src_len , lengths
        score =  amr_len x  batch x src_len , lengths
        roots = batch

       total_loss,total_data
    '''
    probBatch,posteriors_likelihood_score = probs
    posterior,likeli,score = posteriors_likelihood_score
    if  opt.jamr :
        out = posterior.data*likeli.data
        p = out.sum(1)+epsilon
        total_loss = -( torch.log(p) ).sum()
    elif opt.renyi_alpha > 0:
        out = alginBatch.data*(posterior.data+epsilon).pow(opt.renyi_alpha)*(likeli.data+epsilon).pow(1-opt.renyi_alpha)
        p = out.sum(1)+epsilon
        total_loss = torch.log(p).sum()/(opt.renyi_alpha-1)
    else: #hierachical relaxation
        out = alginBatch.data*posterior.data*likeli.data
        p = torch.nn.functional.relu(out.sum(1))+epsilon
        total_loss = -( torch.log(p) ).sum()
    assert not np.isnan(total_loss.data.cpu().numpy()).any(),("concept\n",epoch,posterior.data,likeli.data)
    if opt.jamr  :
        posterior_loss = Variable(torch.zeros(1).cuda())
    else:
        posterior_loss = opt.sink_re*posterior_regularizor(posterior) if  opt.sink_re != 0 else Variable(torch.zeros(1).cuda())
        if opt.prior_t :
            posterior_loss = posterior_loss + sinkhorn_score_regularizor(score)
    assert not np.isnan(posterior_loss.data.cpu().numpy()).any(), ("posterior\n",epoch,posterior.data )
    total_data = out.size(0)
    if rel_roles_batch:
        probBatch,rel_prob = probBatch
        root_loss = - sum(log_soft_max[i] for log_soft_max,i in zip(roots_log_soft_max,gold_roots))
        assert not np.isnan(root_loss.data.cpu().numpy()).any(), ("root_loss\n",epoch,root_loss.data )
        total_roots =  len(gold_roots)  #- (rel_roles_batch.data.data == 0 ).sum()
        total_rel =  len(rel_roles_batch[0].data)  #- (rel_roles_batch.data.data == 0 ).sum()
        rel_loss =  torch.nn.functional.nll_loss(rel_prob.data,rel_roles_batch.data,size_average = False)
        assert not np.isnan(rel_loss.data.cpu().numpy()).any(), ("srlloss\n",epoch,rel_loss.data )
        total_loss = ([total_loss,posterior_loss],root_loss, rel_loss)
        total_data = (total_data, total_roots,total_rel )
        return total_loss,total_data
    else:
        return ([total_loss,posterior_loss]),total_data



def eval(model,AmrDecoder,data,dicts,rel=False):

    concept_scores = concept_score_initial(dicts)

    rel_scores = rel_scores_initial()

    model.eval()
    for batchIdx in range(len(data)):

        order, srcBatch,tgtBatch,alginBatch,rel_batch,rel_index_batch,rel_roles_batch,gold_roots,sourceBatch = data[batchIdx]
        probBatch = model(srcBatch,rel=False)


        amr_pred_seq,concept_batches,aligns,dependent_mark_batch = AmrDecoder.probAndSourceToAmr(sourceBatch,srcBatch,probBatch,getsense= opt.get_sense and not rel)
        if rel:
            rel_batch_data,align_batch = rel_to_batch(concept_batches,aligns,data,dicts)

            rel_roles_prob,roots_log_soft_max = model((rel_batch_data,srcBatch,align_batch),rel=True)
            graphs,rel_triples  =  AmrDecoder.relProbAndConToGraph(concept_batches,rel_roles_prob,roots_log_soft_max,(dependent_mark_batch,aligns),opt.get_sense)
            if opt.get_sense:
                concept_batches = AmrDecoder.graph_to_concepts_batches(graphs)
            for score_h in rel_scores:
                if score_h.second_filter:
                    t,p,tp = score_h.T_P_TP_Batch(rel_triples,list(zip(*sourceBatch))[5],second_filter_material = (concept_batches,list(zip(*sourceBatch))[4]))
                else:
                    t,p,tp = score_h.T_P_TP_Batch(rel_triples,list(zip(*sourceBatch))[5])


        if batchIdx < 1:
            print ("")
            print ("source",sourceBatch[-1][:4])
            print ("")
            print ("pre seq",amr_pred_seq[-1])
            print ("")
            print ("pre srlseq",concept_batches[-1])
            print ("")
            print ("re-ca gold",sourceBatch[-1][7])



            if rel:

                print ("")
                print ("pre triples",[quandraple[:-1] for quandraple in rel_triples[-1]])
                print ("")
                print("gold seq",sourceBatch[-1][4])
                print("gold triples",sourceBatch[-1][5])
                print ("")
                print ("")



        for score_h in concept_scores:
            t,p,tp = score_h.T_P_TP_Batch(concept_batches,list(zip(*sourceBatch))[4])

    model.train()
    return concept_scores,rel_scores

def trainModel(model,AmrDecoder, trainData, validData, dicts, optim,best_f1 = 0 ,best_epoch = 0,always_print = False):
    model.train()
    AmrDecoder.train()
    start_time = time.time()
    def trainEpoch(epoch):
        total_loss, report_loss = 0, 0
        posterior_total_loss, posterior_report_loss = 0, 0
        total_words, report_words = 0, 0

        root_total_loss, root_report_loss = 0, 0
        root_total_words, root_report_words = 0, 0

        rel_total_loss, rel_report_loss = 0, 0
        rel_total_words, rel_report_words = 0, 0


        start = time.time()

        bs = 99999
        for i in range(min(len(trainData),bs)):

            model.zero_grad()

            order,srcBatch,tgtBatch,alginBatch,rel_batch,rel_index_batch,rel_roles_batch,gold_roots,sourceBatch =trainData[i]

            probBatch,posteriors_likelihood_score,src_enc = model((srcBatch,tgtBatch,alginBatch ),rel=False)

            if opt.rel:
                rel_prob,roots = model((rel_batch,rel_index_batch,srcBatch, posteriors_likelihood_score[0]),rel=True)

                out =  (probBatch,rel_prob),posteriors_likelihood_score
                loss,num_data = Total_Loss(out,tgtBatch,alginBatch,epoch,rel_roles_batch,roots,gold_roots)
            else:
                out = probBatch,posteriors_likelihood_score
                loss,num_data = Total_Loss(out,tgtBatch,alginBatch,epoch)





            if len(loss)>2:
                loss,root_loss,rel_loss = loss
                num_data,num_root,num_rel = num_data

                rel_report_loss += rel_loss.data[0]
                rel_total_loss += rel_loss.data[0]

                rel_total_words += num_rel
                rel_report_words += num_rel

                root_total_loss += root_loss.data[0]
                root_report_loss += root_loss.data[0]

                root_total_words += num_root

                root_report_words += num_root


            loss,posterior_loss = loss
            posterior_loss = posterior_loss
            loss = loss + posterior_loss
            report_loss += loss.data[0]
            total_loss += loss.data[0]
            if opt.prior_t:
                posterior_report_loss+= posterior_loss.data[0]
                posterior_total_loss += posterior_loss.data[0]

            total_words += num_data
            report_words += num_data

            if opt.rel:
                loss = loss/num_data+root_loss/num_root+  opt.rel*rel_loss/num_data#*epoch/opt.epochs
            else:
                loss = loss/num_data


            loss.backward()
            grad_norm = optim.step()


     #       del out,out_grad,x
            if i % int(len(trainData)/opt.log_per_epoch) == 0 and i > 0 :
        #        print ("trainModel",srcBatch.size())
                print("Epoch " + str(epoch)+" "+ str(i) +"//"+ str(len(trainData)))
                print("concept loss", 1.0*report_loss/report_words)
          #      if opt.prior_t:
             #       print("posterior loss", posterior_report_loss/report_words)
                print ("grad_norm",grad_norm)
                if rel_report_words > 0 :
                    if rel_report_loss/rel_report_words<50:
                        print("rel perplexity", math.exp(1.0*rel_report_loss/rel_report_words) )
                    else:
                        print("rel loss", rel_report_loss/rel_report_words )

                if root_report_words > 0 :
                    if root_report_loss/root_report_words<50:
                        print("root  perplexity", math.exp(1.0*root_report_loss/root_report_words) )
                    else:
                        print("rootloss", root_report_loss/root_report_words )

                print("tokens/s", 1.0*report_words/(time.time()-start))
                print(str(time.time()-start_time) +" s elapsed")
                
                report_loss = report_words = 0
                rel_report_loss = rel_report_words = 0
                arg_rel_report_loss = arg_rel_report_words = 0

                start = time.time()



        return ((total_loss / total_words,posterior_total_loss/total_words),root_total_loss/root_total_words if root_total_words > 0 else 0,rel_total_loss/rel_total_words if rel_total_words > 0 else 0)
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        for name, param in model.named_parameters():
            assert not np.isnan(np.sum(param.data.cpu().numpy())),(name,"\n",param)
        #  (1) train for one epoch on the training set
        trainData.shuffle()
        train_loss = trainEpoch(epoch)

        concept_loss,posterior_loss = train_loss[0]
        print('Training loss: %g' % concept_loss)
        if opt.prior_t:
            print('Posterior_ppl loss: ' , posterior_loss)



        if train_loss[1] != 0:
            train_ppl = math.exp(train_loss[1]) if train_loss[1]<50 else math.nan
            print('Root Training perplexity: %g' % train_ppl  )

            train_ppl = math.exp(train_loss[2]) if train_loss[2]<50 else math.nan
            print('Role Training perplexity: %g' % train_ppl  )

        print("")

        #  (2) evaluate on the validation set
        validData.shuffle()
        concept_scores,rel_scores= eval(model,AmrDecoder,validData,dicts,rel=opt.rel)


        if opt.rel:
            for score_h in concept_scores[:2]:
                print("")
                print(score_h)
            for score_h in rel_scores:
                print("")
                print(score_h)
        else:
            for score_h in concept_scores:
                print("")
                print(score_h)


        if epoch % 10 == 0:
            print (opt)
        #  (3) maybe update the learning rate
        if opt.optim == 'sgd':
            optim.updateLearningRate(concept_loss, epoch)

        f1 = P_R_F1(*[a+b for (a,b) in zip(rel_scores[1].t_p_tp,concept_scores[1].t_p_tp)])[2] if opt.rel else P_R_F1(*concept_scores[0].t_p_tp)[2]

        if (f1 > best_f1):
            #  (4) drop a checkpoint
            checkpoint = {
                'model': model,
                'opt': opt,
                'epoch': epoch,
                'optim': optim,
            }
            best_f1 = f1
            best_epoch = epoch
            print ("saving best",opt.save_to+opt_str+'valid_best.pt')
            torch.save(checkpoint,
                       opt.save_to+opt_str+'valid_best.pt' )
        print('Validation F1: %g' % f1)
        print('Best Validation F1: %g' % best_f1)
        print("Epoch: ", best_epoch)

    checkpoint = {
        'model': model,
        'opt': opt,
        'epoch': epoch,
        'optim': optim,
    }

    print("\nEvaluate Training")
    concept_scores,rel_scores= eval(model,AmrDecoder,trainData,dicts)
    for score_h in concept_scores:
        print("")
        print(score_h)
    if opt.rel:
        for score_h in rel_scores:
            print("")
            print(score_h)


    p_r_f1 = P_R_F1(*[a+b for (a,b) in zip(rel_scores[1].t_p_tp,concept_scores[1].t_p_tp)]) if opt.rel else P_R_F1(*concept_scores[0].t_p_tp)

    print ("Training Total Precesion, recall, f1",p_r_f1)
    print ("saving last")
    torch.save(checkpoint,
               opt.save_to+opt_str+'last.pt' )



def embedding_from_dicts( opt,dicts):

    embs = dict()

    def initial_embedding():
        print("loading fixed word embedding")

        word_dict = dicts["word_dict"]
        lemma_dict = dicts["lemma_dict"]
        word_initialized = 0
        lemma_initialized = 0
        with open(embed_path, 'r') as f:
            for line in f:
                parts = line.rstrip().split()
                if len(parts)==2:
                    word_embedding = nn.Embedding(dicts["word_dict"].size(),
                                                  int(parts[1]),
                                          padding_idx=PAD)
                    lemma_lut = nn.Embedding(dicts["lemma_dict"].size(),
                                              opt.lemma_dim,
                                              padding_idx=PAD)
                    print (parts)
                else:
                    id,id2 = word_dict[parts[0]],lemma_dict[parts[0]]
                    if id != UNK and id < word_embedding.num_embeddings:
                        tensor = torch.FloatTensor([float(s) for s in parts[-word_embedding.embedding_dim:]]).type_as(word_embedding.weight.data)
                        word_embedding.weight.data[id].copy_(tensor)
                        word_initialized += 1

                    if False and id2 != UNK and id2 < lemma_lut.num_embeddings :
                        tensor = torch.FloatTensor([float(s) for s in parts[-lemma_lut.embedding_dim:]]).type_as(lemma_lut.weight.data)
                        lemma_lut.weight.data[id2].copy_(tensor)
                        lemma_initialized += 1

        print ("word_initialized", word_initialized)
        print ("lemma initialized", lemma_initialized)
        print ("word_total", word_embedding.num_embeddings)
        return word_embedding,lemma_lut
  #  print (dicts["sensed_dict"])
    if opt.initialize_word:
        word_fix_lut,lemma_lut = initial_embedding()
        word_fix_lut.requires_grad = False      ##need load
    else:
        word_fix_lut =  nn.Embedding(dicts["word_dict"].size(),
                                                  300,
                                          padding_idx=PAD)
        lemma_lut = nn.Embedding(dicts["lemma_dict"].size(),
                                  opt.lemma_dim,
                                  padding_idx=PAD)

    high_lut = nn.Embedding(dicts["high_dict"].size(),
                              opt.lemma_dim,
                              padding_idx=PAD)

    pos_lut = nn.Embedding(dicts["pos_dict"].size(),
                                  opt.pos_dim,
                                  padding_idx=PAD)
    ner_lut = nn.Embedding(dicts["ner_dict"].size(),
                              opt.ner_dim,
                              padding_idx=PAD)


    rel_lut =  nn.Embedding(dicts["rel_dict"].size(),
                              1)  #not actually used, but handy to pass number of relations
    cat_lut = nn.Embedding(dicts["category_dict"].size(),
                              opt.cat_dim,
                              padding_idx=PAD)


    aux_lut = nn.Embedding(dicts["aux_dict"].size(),
                              1,
                              padding_idx=PAD)

    if opt.cuda:
        word_fix_lut.cuda()
        lemma_lut.cuda()
        pos_lut.cuda()
        ner_lut.cuda()
        cat_lut.cuda()
        aux_lut.cuda()
        high_lut.cuda()

    embs["word_fix_lut"] = word_fix_lut
    embs["aux_lut"] = aux_lut
    embs["high_lut"] = high_lut
    embs["lemma_lut"] = lemma_lut
    embs["pos_lut"] = pos_lut
    embs["ner_lut"] = ner_lut
    embs["cat_lut"] = cat_lut
    embs["rel_lut"] = rel_lut


    return embs


def main():
    dicts = read_dicts()
    print('Building model...')
    AmrDecoder = parser.AMRProcessors.AMRDecoder(opt,dicts)

    with_jamr = "_with_jamr" if opt.jamr else "_without_jamr"
    suffix = ".pickle"+with_jamr+"_processed"
    trainFolderPath = opt.folder+"/training/"
    trainingFilesPath = folder_to_files_path(trainFolderPath,suffix)

    devFolderPath = opt.folder+"/dev/"
    devFilesPath = folder_to_files_path(devFolderPath,suffix)

    testFolderPath = opt.folder+"/test/"
    testFilesPath = folder_to_files_path(testFolderPath,suffix)

    dev_data = DataIterator(devFilesPath,opt,dicts["rel_dict"] ,volatile = True)


    training_data = DataIterator(trainingFilesPath,opt,dicts["rel_dict"])
    f1 = 0
    if opt.train_from is None:
        embs = embedding_from_dicts( opt,dicts)

        AmrModel = parser.models.AmrModel(opt, embs)
        print(AmrModel)
        parameters_to_train = []
        for p in AmrModel.parameters():
            if p.requires_grad:
                parameters_to_train.append(p)
    else:
        AmrModel,parameters_to_train,optt =  load_old_model(dicts,opt)
        opt.start_epoch =  1

        concept_scores,rel_scores= eval(AmrModel,AmrDecoder, dev_data,dicts,rel=False)
        for score_h in concept_scores:
            print("")
            print(score_h)

        for score_h in rel_scores:
            print("")
            print(score_h)

        f1 = P_R_F1(*[a for (a,b) in zip(rel_scores[1].t_p_tp,concept_scores[1].t_p_tp)])[2] if opt.rel else P_R_F1(*concept_scores[0].t_p_tp)[2]
        print ("best_f1",f1)

    optim = parser.Optim.Optim(
        parameters_to_train, opt.optim, opt.learning_rate, opt.max_grad_norm,
        lr_decay=opt.learning_rate_decay,
        start_decay_at=opt.start_decay_at,
        weight_decay=opt.weight_decay
    )
    print(' * number of training sentences. %d' %
          len(training_data.src))
    print(' * batch  size.. %d' % opt.batch_size)

    nParams = sum([p.nelement() for p in AmrModel.parameters()])
    print('* number of parameters: %d' % nParams)

    trainModel(AmrModel, AmrDecoder,training_data,  dev_data, dicts, optim,best_f1=f1)


if __name__ == "__main__":
    global opt
    global opt_str
    opt = train_parser().parse_args()

    with_jamr = "_with_jamr" if opt.jamr else "_without_jamr"

    opt.lemma_dim = opt.dim
    opt.high_dim = opt.dim

    #used for deciding saved model name
    opt_str =  "gpus_"+str(opt.gpus[0])


    opt.cuda = len(opt.gpus)
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with -cuda")

    if opt.cuda:
        cuda.set_device(opt.gpus[0])
    main()
