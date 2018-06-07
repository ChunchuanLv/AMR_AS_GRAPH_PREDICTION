

import torch
import torch.nn as nn
def freeze(m,t=0):
    if isinstance(m,nn.Dropout):
        m.p  = t
    m.dropout =t


from copy import deepcopy
def load_old_model(dicts,opt,generate=False):
    model_from = opt.train_from
    print('Loading from checkpoint at %s' % model_from)
    if opt.gpus[0] != -1:
        print ('from model in gpus:'+str(opt.from_gpus[0]),' to gpu:'+str(opt.gpus[0]))
        checkpoint = torch.load(model_from, map_location={'cuda:'+str(opt.from_gpus[0]): 'cuda:'+str(opt.gpus[0])})
    else:
        print ('from model in gpus:'+str(opt.from_gpus[0]),'to cpu ')
        checkpoint = torch.load(model_from, map_location={'cuda:'+str(opt.from_gpus[0]): 'cpu'})
    print("Model loaded")
    optt = checkpoint["opt"]
    rel = optt.rel
    AmrModel = checkpoint['model']
    if optt.rel == 1:
        if not opt.train_all:
            AmrModel.concept_decoder = deepcopy(AmrModel.concept_decoder)
            for name, param in AmrModel.concept_decoder.named_parameters():
                param.requires_grad = False
            AmrModel.concept_decoder.apply(freeze)

        parameters_to_train = []
        for name, param in AmrModel.named_parameters():
            if name == "word_fix_lut" or param.size(0) == len(dicts["word_dict"]):
                param.requires_grad = False
            if param.requires_grad:
                parameters_to_train.append(param)
        print (AmrModel)
        print ("training parameters: "+str(len(parameters_to_train)))
        return AmrModel,parameters_to_train,optt

    optt.rel = opt.rel
    if opt.rel and not rel  :
        if opt.jamr == 0:
            AmrModel.poserior_m.align_weight = 1
        AmrModel.concept_decoder.apply(freeze)
        opt.independent = True
        AmrModel.start_rel(opt)
        embs = AmrModel.embs
        embs["lemma_lut"].requires_grad = False      ##need load
        embs["pos_lut"].requires_grad = False
        embs["ner_lut"].requires_grad = False
        embs["word_fix_lut"].requires_grad = False
        embs["rel_lut"] =   nn.Embedding(dicts["rel_dict"].size(),
                          opt.rel_dim)
        for param in AmrModel.concept_decoder.parameters():
            param.requires_grad = False
    if not generate and opt.jamr == 0:
        AmrModel.poserior_m.posterior.ST = opt.ST
        AmrModel.poserior_m.posterior.sink = opt.sink
        AmrModel.poserior_m.posterior.sink_t = opt.sink_t

    if opt.cuda:
        AmrModel.cuda()
    else:
        AmrModel.cpu()

    if not generate and opt.jamr == 0:
        if opt.train_posterior:
            for param in AmrModel.poserior_m.parameters():
                param.requires_grad = True
            AmrModel.poserior_m.apply(lambda x: freeze(x,opt.dropout))
        else:
            opt.prior_t = 0
            opt.sink_re = 0
            for param in AmrModel.poserior_m.parameters():
                param.requires_grad = False
    parameters_to_train = []
    if opt.train_all:
        for name, param in AmrModel.named_parameters():
            if name != "word_fix_lut":
                param.requires_grad = True
                parameters_to_train.append(param)
            else:
                print ("not updating "+name)

    else:
        if opt.rel:
            for param in AmrModel.concept_decoder.parameters():
                if   param.requires_grad:
                    param.requires_grad = False
                    print("turing off concept model:  ",param)
            for name,p in AmrModel.named_parameters():
                if name == "word_fix_lut" or p.size(0) == len(dicts["word_dict"]):
                    p.requires_grad = False
                if p.requires_grad:
                    parameters_to_train.append(p)
        else:
            print ([p.size() for p in AmrModel.concept_decoder.parameters()])
            AmrModel.apply(freeze)
            for p in AmrModel.concept_decoder.parameters():
                p.requires_grad = True
                parameters_to_train.append(p)
    print (AmrModel)
    print ("training parameters: "+str(len(parameters_to_train)))
    return AmrModel,parameters_to_train,optt
