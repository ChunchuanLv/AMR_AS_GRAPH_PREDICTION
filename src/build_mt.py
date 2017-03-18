from utility.constants import *
from utility.amr import *
from utility.reader import *
from npmt.rules import *
from utility.pickle_helper import *
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


def add_one_concept(lemma_token,lemma,is_frame):
    for tokens in lemma_token:
        for t in tokens:
            if checkMatch(t,lemma,matches):
                rl.add_lemma_freq(t,lemma,is_frame)

def try_add_lemmatize_c(rl,non_rule_set,c):
    
    if c.is_frame():
        c_t = re.sub(c.RE_FRAME_NUM,"",c.__str__())
    else:
        c_t = c.__str__()
    lemma = c_t
    freq =  non_rule_set[c][0]
    lemma_token = non_rule_set[c][1]
    add_one_concept(lemma_token,lemma,c.is_frame())
    
def try_add_lemmatize_cheat(rl,non_rule_set,threshold):
    executor = ThreadPoolExecutor(max_workers)
    for c in non_rule_set.keys():
        
    #    freq =  non_rule_set[c][0]
        if not c.is_constant() :
            executor.submit( try_add_lemmatize_c, rl,non_rule_set,c )
    executor.shutdown()    
    rl.build_lemma_cheat()
            
    

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

def handle_sentence(snt_token,lemma,amr_t,n,modify=False):
    
    if n % 500 == 0:
        print (n)
    n = n % max_workers
    amr = AMR(amr_t)
    add_count(non_rule_set,rl.get_none_rule(snt_token,amr,lemma),lemma)
    if modify:
        rl.get_matched_concepts(snt_token,amr,lemma)
    
def readFile(filepath,modify=False):
    executor = ThreadPoolExecutor(max_workers)
    
    with open(filepath.replace(".txt",".json"),'r') as data_file:    
        all_data = json.load(data_file)
    all_core_exceptions = []
    all_non_core_exceptions = []

    n = 0
    for data in all_data:
        n=n+1
        snt_token = data["snt_token"] 
        lemma = data["lemma"] 
        amr_t = data["amr_t"]
        handle_sentence(snt_token,lemma,amr_t,n,modify)
        #executor.submit( handle_sentence, snt_token,lemma,amr_t,n,modify)
    executor.shutdown()
    return n



# Creating ReUsable Object
rl = rules()

#initializer = lasagne.init.Uniform()
non_rule_set = dict()

#lemmas_to_concept =  read_resource_files( f_r.get_frames())
for filepath in trainingFilesPath:
    print(("reading "+filepath.split("/")[-1]+"......"))
    n = readFile(filepath,modify=True)
    print(("done reading "+filepath.split("/")[-1]+", "+str(n)+" sentences processed"))
#non_rule_set = non_rule_set_last
text_num,high_non_text,low_non_text=unmixe(non_rule_set,threshold )
print ("initial",threshold,len(non_rule_set),len(text_num),len(high_non_text),len(low_non_text))
#print (len(concept_embedding))

#10 unbroken 13276 6241 1806 5229   
#10 unbroken+rematch 13071 6241 2175 4655 
#2 broken 13273 6367 3612 3294
#10 broken 13273 6367 1785 5121
#10 unbroken 13273 6367 1785 5121  
#10 broken-rematch 12925 6367 2097 4461
#10 unbroken+rematch 13148 6367 2056 4725
#50 unbroken 50 11538 6455 200 4883
#50 unbroken+rematch+combine_num  50 10332 6463 144 3725
            #50 unbroken+rematch  50 10351 6455 145 3751


#118936, 169, 3955, 85, 2549, 978

matches = [0,0,0,0,0,0,0,0]


non_rule_set_last = non_rule_set
try_add_lemmatize_cheat(rl,non_rule_set_last,threshold)
#436 78 148 210

print (matches)
#441 78 151 212

#print (rl.lemmatize_cheat)

save_rule(rl,"data/rule_f")

non_rule_set = dict()

#lemmas_to_concept =  read_resource_files( f_r.get_frames())
for filepath in trainingFilesPath:
    print(("reading "+filepath.split("/")[-1]+"......"))
    n = readFile(filepath)
    print(("done reading "+filepath.split("/")[-1]+", "+str(n)+" sentences processed"))
#non_rule_set = non_rule_set_last
text_num,high_non_text,low_non_text=unmixe(non_rule_set,threshold )
print ("rematched",threshold,len(non_rule_set),len(text_num),len(high_non_text),len(low_non_text))
  


non_rule_set_f = Pickle_Helper("data/non_rule_set") 
non_rule_set_f.dump(non_rule_set_last,"initial_non_rule_set")
non_rule_set_f.dump(non_rule_set,"non_rule_set")
non_rule_set_f.save()

