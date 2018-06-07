#!/usr/bin/env python3.6
# coding=utf-8
'''

Scripts to run the model over preprocessed data to generate evaluatable results

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

from parser.DataIterator import DataIterator,rel_to_batch
import parser
import torch
from torch import cuda
from utility.Naive_Scores import *
from parser.AMRProcessors import graph_to_amr
from utility.data_helper import folder_to_files_path

from src.train import read_dicts,load_old_model,train_parser

def generate_parser():
    parser = train_parser()
    parser.add_argument('-output', default="_generate")
    parser.add_argument('-with_graphs', type=int,default=1)
    return parser



def generate_graph(model,AmrDecoder, data_set,dicts,file):

    concept_scores = concept_score_initial(dicts)

    rel_scores = rel_scores_initial()

    model.eval()
    AmrDecoder.eval()
    output = []
    gold_file = []
    for batchIdx in range(len(data_set)):
        order,srcBatch,_,_,_,_,_,gold_roots,sourceBatch =data_set[batchIdx]

        probBatch = model(srcBatch )



        amr_pred_seq,concept_batches,aligns_raw,dependent_mark_batch = AmrDecoder.probAndSourceToAmr(sourceBatch,srcBatch,probBatch,getsense = opt.get_sense )

        amr_pred_seq = [ [(uni.cat,uni.le,uni.aux,uni.sense,uni)  for uni in seq ] for  seq in amr_pred_seq ]


        rel_batch,aligns = rel_to_batch(concept_batches,aligns_raw,data_set,dicts)
        rel_prob,roots = model((rel_batch,srcBatch,aligns),rel=True)
        graphs,rel_triples  =  AmrDecoder.relProbAndConToGraph(concept_batches,rel_prob,roots,(dependent_mark_batch,aligns_raw),opt.get_sense,opt.get_wiki)
        batch_out = [0]*len(graphs)
        for score_h in rel_scores:
            if score_h.second_filter:
                t,p,tp = score_h.T_P_TP_Batch(rel_triples,list(zip(*sourceBatch))[5],second_filter_material =  (concept_batches,list(zip(*sourceBatch))[4]))
            else:
                t,p,tp = score_h.T_P_TP_Batch(rel_triples,list(zip(*sourceBatch))[5])
        for score_h in concept_scores:
            t,p,tp = score_h.T_P_TP_Batch(concept_batches,list(zip(*sourceBatch))[4])
        for i,data in enumerate(zip( sourceBatch,amr_pred_seq,concept_batches,rel_triples,graphs)):
            source,amr_pred,concept, rel_triple,graph= data
            predicated_graph = graph_to_amr(graph)

            out = []
            out.append( "# ::tok "+" ".join(source[0])+"\n")
            out.append(  "# ::lem "+" ".join(source[1])+"\n")
            out.append(  "# ::pos "+" ".join(source[2])+"\n")
            out.append(  "# ::ner "+" ".join(source[3])+"\n")
            out.append(  "# ::predicated "+" ".join([str(re_cat[-1]) for re_cat in amr_pred])+"\n")
            out.append(  "# ::transformed final predication "+" ".join([str(c) for c in concept])+"\n")
            out.append( AmrDecoder.nodes_jamr(graph))
            out.append( AmrDecoder.edges_jamr(graph))
            out.append( predicated_graph)
            batch_out[order[i]] = "".join(out)+"\n"
        output += batch_out
    t_p_tp = list(map(lambda a,b:a+b, concept_scores[1].t_p_tp,rel_scores[1].t_p_tp))
    total_out = "Smatch"+"\nT,P,TP: "+ " ".join([str(i) for i in  t_p_tp])+"\nPrecesion,Recall,F1: "+ " ".join([str(i)for i in  P_R_F1(*t_p_tp)])
    print(total_out)
    for score_h in rel_scores:
        print("")
        print(score_h)
    file = file.replace(".pickle",".txt")
    with open(file+ opt.output, 'w+') as the_file:
        for data in output:
            the_file.write(data+'\n')
    print(file+ opt.output+" written.")
    return concept_scores,rel_scores,output


def main(opt):
    dicts = read_dicts()
    assert opt.train_from
    with_jamr = "_with_jamr" if opt.jamr else "_without_jamr"
    suffix = ".pickle"+with_jamr+"_processed"
    trainFolderPath = opt.folder+"/training/"
    trainingFilesPath = folder_to_files_path(trainFolderPath,suffix)

    devFolderPath = opt.folder+"/dev/"
    devFilesPath = folder_to_files_path(devFolderPath,suffix)

    testFolderPath = opt.folder+"/test/"
    testFilesPath = folder_to_files_path(testFolderPath,suffix)



    AmrDecoder = parser.AMRProcessors.AMRDecoder(opt,dicts)
    AmrDecoder.eval()
    AmrModel,parameters,optt =  load_old_model(dicts,opt,True)
    opt.start_epoch =  1

    out = "/".join(testFilesPath[0].split("/")[:-2])+ "/model"
    with open(out, 'w') as outfile:
        outfile.write(opt.train_from+"\n")
        outfile.write(str(AmrModel)+"\n")
        outfile.write(str(optt)+"\n")
        outfile.write(str(opt))

    print('processing testing')
    for file in testFilesPath:
        dev_data = DataIterator([file],opt,dicts["rel_dict"],volatile = True)
        concept_scores,rel_scores,output =generate_graph(AmrModel,AmrDecoder,dev_data,dicts,file)

    print('processing validation')
    for file in devFilesPath:
        dev_data = DataIterator([file],opt,dicts["rel_dict"],volatile = True)
        concept_scores,rel_scores,output =generate_graph(AmrModel,AmrDecoder,dev_data,dicts,file)



    print('processing training')
    for file in trainingFilesPath:
        dev_data = DataIterator([file],opt,dicts["rel_dict"],volatile = True)
        concept_scores,rel_scores,output =generate_graph(AmrModel,AmrDecoder,dev_data,dicts,file)


if __name__ == "__main__":
    print ("      ")
    print ("      ")
    global opt
    opt = generate_parser().parse_args()
    opt.lemma_dim = opt.dim
    opt.high_dim = opt.dim

    opt.cuda = len(opt.gpus)

    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with -cuda")

    if opt.cuda and opt.gpus[0] != -1:
        cuda.set_device(opt.gpus[0])
    main(opt)