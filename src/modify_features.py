from utility.constants import *
from utility.amr import *
from utility.rules import *
from utility.converter import *
from utility.reader import *
from utility.pickle_helper import *
from numpy import array
import json
import codecs
import sys
import os
from nltk.corpus import wordnet
import spacy

nlp = spacy.load('en')
import threading

from concurrent.futures import *
max_workers=24

tag_locks = []
for i in range(max_workers):
    tag_locks.append(threading.Lock())


# Creating ReUsable Object

lock = threading.Lock()

def add_seq_to_dict(dictionary,seq):
    for i in seq:
        if i not in dictionary:
            dictionary[i] = len(dictionary)
            
def seq_to_id(dictionary,seq):
    id_seq = []
    for i in seq:
        if i not in dictionary:
            id_seq.append(0)
        else:
            id_seq.append(dictionary[i])
    return id_seq
            
def amr_seq_to_seq(amr_seq):
    seq = []
    for l in amr_seq:
        for t in l:
            seq.append(t[0].__repr__())
    return seq

def amr_seq_to_id(dictionary,amr_seq):
    id_seq = []
    for l in amr_seq:
        id_l = []
        for t in l:
            if t[0] in dictionary:
                id_l.append((dictionary[t[0]],t[1]))
            else:
                id_l.append((0,t[1]))
        id_seq.append(id_l)
    return id_seq
        
def handle_sentence(data,filepath,training,n):
    
    if n % 100 == 0:
        print (filepath,n)

    ner = data["ner"]
    snt_token = data["snt_token"]
    pos = data["pos"]
    lemma =  data["lemma"] 
    amr_t = data["amr_t"]
    amr = AMR(amr_t)
    amr_seq = amr_to_seq(amr,snt_token,lemma,rl,high_freq)
    data["amr_seq"] = amr_seq_to_seq(amr_seq   )
    
    if training:
        add_seq_to_dict(word_to_id,snt_token)
        add_seq_to_dict(lemma_to_id,lemma)
        add_seq_to_dict(pos_to_id,pos)
        add_seq_to_dict(ner_to_id,ner)         
        add_seq_to_dict(concept_to_id,data["amr_seq"])
        
    data["pos_id"] = seq_to_id(pos_to_id,pos)
    data["snt_id"] = seq_to_id(word_to_id,snt_token)
    data["lemma_id"] = seq_to_id(lemma_to_id,lemma)
    data["ner_id"] = seq_to_id(ner_to_id,ner)
    
    data["amr_id"] = amr_seq_to_id(concept_to_id,amr_seq)
            
    
def readFile(filepath,training = True):

    
    with open(filepath.replace(".txt",".json"),'r') as data_file:    
        all_data = json.load(data_file)
    n = 0
    for data in all_data:
        n = n+1
        handle_sentence(data,filepath,training,n)
    
    with open(filepath.replace(".txt",".json"), 'w+') as outfile:
        json.dump(all_data, outfile)
    return len(all_data)
import itertools as it
merge = lambda *args: dict(it.chain.from_iterable(it.imap(dict.iteritems, args)))

rl = load_rule("data/rule_f")

non_rule_set_f = Pickle_Helper("data/non_rule_set") 
non_rule_set = non_rule_set_f.load()["non_rule_set"]
threshold = 10
high_text_num,high_non_text,low_non_text=unmixe(non_rule_set,threshold)
high_freq = dict(list(high_non_text.items()) + list(high_text_num.items()))
print(len(high_non_text),len(high_text_num),len(high_freq))
word_to_id = dict()
word_to_id["_UNK"] = 0
lemma_to_id = dict()
lemma_to_id["_UNK"] = 0
pos_to_id = dict()
pos_to_id["_UNK"] = 0
ner_to_id = dict()
ner_to_id["_UNK"] = 0
concept_to_id = dict()
concept_to_id[AMRConstant("UNK").__repr__()] = 0

with open("data/word_to_id.json",'w+') as outfile:    
    json.dump(word_to_id,outfile)
with open("data/lemma_to_id.json",'w+') as outfile:    
    json.dump(lemma_to_id,outfile)
with open("data/pos_to_id.json",'w+') as outfile:    
    json.dump(pos_to_id,outfile)
with open("data/ner_to_id.json",'w+') as outfile:    
    json.dump(ner_to_id,outfile)
with open("data/concept_to_id.json",'w+') as outfile:    
    json.dump(concept_to_id,outfile)
#lemmas_to_concept =  read_resource_files( f_r.get_frames())
for filepath in trainingFilesPath:
    print(("reading "+filepath.split("/")[-1]+"......"))
    n = readFile(filepath)
    print(("done reading "+filepath.split("/")[-1]+", "+str(n)+" sentences processed"))
    
    
    
with open("data/word_to_id.json",'w+') as outfile:    
    json.dump(word_to_id,outfile)
with open("data/lemma_to_id.json",'w+') as outfile:    
    json.dump(lemma_to_id,outfile)
with open("data/pos_to_id.json",'w+') as outfile:    
    json.dump(pos_to_id,outfile)
with open("data/ner_to_id.json",'w+') as outfile:    
    json.dump(ner_to_id,outfile)
with open("data/concept_to_id.json",'w+') as outfile:    
    json.dump(concept_to_id,outfile)
    