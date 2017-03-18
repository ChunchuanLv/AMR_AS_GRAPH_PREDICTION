from utility.constants import *
from utility.amr import *
from utility.reader import *
from utility.rules import *
from utility.pickle_helper import *
from utility.converter import *
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from numpy import array
import json
import codecs
import sys
import os
#import lasagne.init
from nltk.metrics.distance import edit_distance
from nltk.tag import StanfordPOSTagger

import threading

from concurrent.futures import *
max_workers=24

st_pos = []
pos_locks = []


def handle_sentence(snt_token,pos,amr_t,n):
    
    if n % 100 == 0:
        print (n)
    n = n % max_workers
    amr = AMR(amr_t)
    # print (snt)
    amr_seq = amr_to_seq(amr,snt_token,pos,rl,high_freq)
    
def readFile(filepath):
    executor = ThreadPoolExecutor(max_workers)
    
    with open(filepath.replace(".txt",".json"),'r') as data_file:    
        all_data = json.load(data_file)
    all_core_exceptions = []
    all_non_core_exceptions = []

    n = 0
    non_core_of_bad = 0
    core_bad = 0
    for data in all_data:
        n=n+1
        snt_token = data["snt_token"] 
        pos =data["pos"]
        amr_t = data["amr"]
        handle_sentence(snt_token,pos,amr_t,n)
        #executor.submit( handle_sentence, snt_token,pos,amr_t,n )
    executor.shutdown()
    return n

def initial_embedding():
    concept_embedding[Concept(TOP)] = initializer((con_dim,))
    concept_embedding[Concept(END)] = initializer((con_dim,))
    lemma_embedding[UNK] = initializer((le_dim,))
    word_embedding[UNK] = lemma_embedding[UNK]
    for i, line in enumerate(open(embed_path, 'r')):
        parts = line.rstrip().split()
        word_embedding[parts[0]] = list(map(float, parts[1:]))
        le = wordnet_lemmatizer.lemmatize(parts[0],'v')
        if le not in lemma_embedding:
            lemma_embedding[le] = initializer((le_dim,))

#add self here and normalize
def turn_text_list_to_index_list(lemmas_to_concept,lemma_dict,concept_dict):
    index_to_index = dict()
    for le, cons in lemmas_to_concept.items():
        print (le,cons.keys())
        le_index = lemma_dict[le]
        print (le_index)
        cons_indexes = sorted([concept_dict[con] for con in cons.keys()])
        index_to_index[le_index] = array(cons_indexes)
    return index_to_index

def add_resource_embed(short_list,lemma_embedding,concept_embedding):
    for k in short_list.keys():
        if k not in lemma_embedding:
            lemma_embedding[k] = initializer((le_dim,))
        for c in short_list[k].keys():
            if c not in concept_embedding:
                concept_embedding[c] = initializer((con_dim,))
word_embedding = dict()
lemma_embedding = dict()
concept_embedding = dict()
short_list = dict()

embeddings_f = Pickle_Helper("data/embeddings") 
non_rule_set_f = Pickle_Helper("data/non_rule_set") 
softmax_short_list = Pickle_Helper("data/softmax_short_list") 
to_id_f= Pickle_Helper("data/to_id_f") 


# Creating ReUsable Object
rl = load_rule("data/rule_f")
non_rule_set = non_rule_set_f.load()["non_rule_set"]
text_num,high_freq,low_frequency=unmixe(non_rule_set,threshold)
print (threshold,len(non_rule_set),len(text_num),len(high_frequency),len(low_frequency))


#lemmas_to_concept =  read_resource_files( f_r.get_frames())
for filepath in trainingFilesPath:
    print(("reading "+filepath.split("/")[-1]+"......"))
    n = readFile(filepath)
    print(("done reading "+filepath.split("/")[-1]+", "+str(n)+" sentences processed"))

print (matches)
  
  

non_rule_set_f.dump(non_rule_set,"non_rule_set")
embeddings_f.save()
to_id_f.save()
non_rule_set_f.save()
softmax_short_list.save()

