#!/usr/bin/env python3.6
# coding=utf-8
'''

Scripts to build StringCopyRules and ReCategorizor

Data path information should also be specified here for
trainFolderPath, devFolderPath and testFolderPath
as we allow option to choose from two version of data.
@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

from utility.StringCopyRules import *
from utility.ReCategorization import *
from utility.data_helper import *


import argparse
def arg_parser():
    parser = argparse.ArgumentParser(description='rule_system_build.py')

    ## Data options
    parser.add_argument('-threshold', default=5, type=int,
                        help="""threshold for non-aligned high frequency concepts""")

    parser.add_argument('-jamr', default=0, type=int,
                        help="""wheather to enhance string matching with additional jamr alignment""")
    parser.add_argument('-suffix', default=".txt_pre_processed", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('-folder', default=allFolderPath, type=str ,
                        help="""the folder""")
    return parser
parser = arg_parser()
opt = parser.parse_args()
threshold = opt.threshold
suffix = opt.suffix + "_jamr" if opt.jamr else opt.suffix
with_jamr = "_with_jamr" if opt.jamr else "_without_jamr"
trainFolderPath = opt.folder+"/training/"
trainingFilesPath = folder_to_files_path(trainFolderPath,suffix)

devFolderPath = opt.folder+"/dev/"
devFilesPath = folder_to_files_path(devFolderPath,suffix)

testFolderPath = opt.folder+"/test/"
testFilesPath = folder_to_files_path(testFolderPath,suffix)


lock = threading.Lock()
def add_count(store,new,additional=None):
    lock.acquire()
    
    for i in new:
        if not i in store:
            store[i] = [1,[additional]]
        else:
            store[i][0] = store[i][0] + 1 
            store[i][1].append(additional)
    lock.release()

def handle_sentence(data,n,update_freq,use_template,jamr = False):
    
    if n % 500 == 0:
        print (n)
    snt_token = data["tok"]
    pos_token = data["pos"]
    lemma_token = data["lem"]
    amr_t = data["amr_t"]
    aligns = data["align"]
    v2c = data["node"]
    amr = AMRGraph(amr_t,aligns=aligns)
    amr.check_consistency(v2c)
    lemma_str  =" ".join(lemma_token)
    if use_template:
        fragment_to_node_converter.match(amr,rl ,snt_token,lemma_token,pos_token,lemma_str,jamr=jamr )
    fragment_to_node_converter.convert(amr,rl ,snt_token,lemma_token,pos_token,lemma_str )
    results =  rl.get_matched_concepts(snt_token,amr,lemma_token,pos_token,with_target=update_freq,jamr=jamr)
    if update_freq:
        for n_c_a in results :
            for i_le in n_c_a[2]:
                rl.add_lemma_freq(i_le[1],n_c_a[1].le,n_c_a[1].cat,sense = n_c_a[1].sense)

    snt_str = " ".join(snt_token)
    none_rule = [n_c_a[1] for n_c_a in results if len(n_c_a[2])==0]
    add_count(non_rule_set,none_rule,snt_str)


def readFile(filepath,update_freq=False,use_template=True):
    all_data = load_text_jamr(filepath)

    with open(filepath.replace(".txt",".tok"),'w') as output_file:
        n = 0
        for data in all_data:
            n=n+1
            snt_token = data["tok"]
            output_file.writelines("\t".join(snt_token))
            if opt.jamr:
                handle_sentence(data,n,update_freq,use_template,jamr=True)
            else:
                handle_sentence(data,n,update_freq,use_template,jamr=False)
    return n



rl = rules()
non_rule_set = dict()
fragment_to_node_converter = ReCategorizor(training=True)
#
non_rule_set_last = non_rule_set
rl.build_lemma_cheat()
#
non_rule_set = dict()
#lemmas_to_concept =  read_resource_files( f_r.get_frames())
for filepath in trainingFilesPath:  #actually already combined into one
    print(("reading "+filepath.split("/")[-1]+"......"))
    n = readFile(filepath,update_freq=True,use_template = True)
    print(("done reading "+filepath.split("/")[-1]+", "+str(n)+" sentences processed"))
#non_rule_set = non_rule_set_last
high_text_num,high_frequency,low_frequency,low_text_num=unmixe(non_rule_set,threshold )
print ("initial converted,threshold,len(non_rule_set),high_text_num,high_frequency,low_frequency,low_text_num")
print ("initial converted",threshold,len(non_rule_set),len(high_text_num),len(high_frequency),len(low_frequency),len(low_text_num))
#print (len(concept_embedding))
#
#
#
non_rule_set_initial_converted = non_rule_set
rl.build_lemma_cheat()
fragment_to_node_converter.save(path="data/graph_to_node_dict_extended"+with_jamr)
fragment_to_node_converter = ReCategorizor(from_file=False, path="data/graph_to_node_dict_extended"+with_jamr,training=False)
rl.save("data/rule_f"+with_jamr)
non_rule_set = dict()
NERS = {}

#need to rebuild copying dictionary again based on recategorized graph
for filepath in trainingFilesPath:
    print(("reading "+filepath.split("/")[-1]+"......"))
    n = readFile(filepath,update_freq=False,use_template=False)
    print(("done reading "+filepath.split("/")[-1]+", "+str(n)+" sentences processed"))

non_rule_set_f = Pickle_Helper("data/non_rule_set")
non_rule_set_f.dump(non_rule_set,"non_rule_set")
non_rule_set_f.save()



#only intermediate data, won't be useful for final parser
non_rule_set_f = Pickle_Helper("data/non_rule_set")
non_rule_set_f.dump(non_rule_set_last,"initial_non_rule_set")
non_rule_set_f.dump(non_rule_set_initial_converted,"initial_converted_non_rule_set")
non_rule_set_f.dump(non_rule_set,"non_rule_set")
non_rule_set_f.save()

high_text_num,high_frequency,low_frequency,low_text_num=unmixe(non_rule_set,threshold )
print ("final converted,threshold,len(non_rule_set),high_text_num,high_frequency,low_frequency,low_text_num")
print ("final converted",threshold,len(non_rule_set),len(high_text_num),len(high_frequency),len(low_frequency),len(low_text_num))