__author__ = 's1544871'
from utility.constants import *
from utility.amr import *
from npmt.rules import *
from utility.converter import *
from utility.reader import *
from utility.pickle_helper import *
from npmt.Dict import *
import json
import sys
import os


def add_seq_to_dict(dictionary,seq):
    for i in seq:
        dictionary.add(i)

def seq_to_id(dictionary,seq):
    id_seq = []
    for i in seq:
        id_seq.append(dictionary[i])
    id_seq.append(EOS)
    return id_seq

#id_seq :  [(con_lemma,cat,rel])]
def amr_seq_to_id(lemma_dict,concept_dict,category_dict,rel_dict,amr_seq):
    id_seq = [[BOS,BOS,BOS]]
    for l in amr_seq:
        id_seq.append([lemma_dict[l[3]],category_dict[l[1]],rel_dict[l[2]]])
    id_seq.append([EOS,EOS,EOS])
    return id_seq

#id_seq :  [(con_lemma,cat,[id],rel])]
def amr_seq_to_index(amr_seq,snt_len):
    index_seq = [[-2]] #just a dumb
    for l in amr_seq:
        index_seq.append([t[2] for t in l[0]])
    index_seq.append([snt_len-1])
    return index_seq


def amr_seq_to_dict(lemma_dict,concept_dict,category_dict,rel_dict,seq):
    for i in seq:
        category_dict.add(i[1])
        rel_dict.add(i[2])
        lemma_dict.add(i[3])
        for t in i[0]:
            if t[1] == HIGH:
                concept_dict.add(i[3],lemma_dict[i[3]])   #concept_dict in lemma_dict
def handle_sentence(data,n):

    if n % 1000 == 0:
        print (n)

    ner = data["ner"]
    snt_token = data["snt_token"]
    pos = data["pos"]
    lemma =  data["lemma"]


    amr_t = data["amr_t"]
    amr = AMR(amr_t)
    amr_seq = fake_amr_to_seq(amr,snt_token,lemma,rl,high_freq)


def readFile(filepath):


    with open(filepath.replace(".txt",".json"),'r') as data_file:
        all_data = json.load(data_file)
    n = 0
    for data in all_data:
        n = n+1
        handle_sentence(data,n)
    return len(all_data)

rl = load_rule("data/rule_f")

non_rule_set_f = Pickle_Helper("data/non_rule_set")
non_rule_set = non_rule_set_f.load()["non_rule_set"]
threshold = 5
high_text_num,high_non_text,low_non_text=unmixe(non_rule_set,threshold)
high_freq = dict(list(high_non_text.items()) + list(high_text_num.items()))
print(len(high_non_text),len(high_text_num),len(high_freq))

print(("processing training set"))
for filepath in trainingFilesPath:
    print(("reading "+filepath.split("/")[-1]+"......"))
    n = readFile(filepath)
    print(("done reading "+filepath.split("/")[-1]+", "+str(n)+" sentences processed"))


print(("processing development set"))
for filepath in devFilesPath:
    print(("reading "+filepath.split("/")[-1]+"......"))
    n = readFile(filepath)
    print(("done reading "+filepath.split("/")[-1]+", "+str(n)+" sentences processed"))


print(("processing test set"))
for filepath in testFilesPath:
    print(("reading "+filepath.split("/")[-1]+"......"))
    n = readFile(filepath)
    print(("done reading "+filepath.split("/")[-1]+", "+str(n)+" sentences processed"))


print(len(high_non_text),len(high_text_num),len(high_freq))
