from utility.constants import *
from utility.amr import *
from utility.reader import *
from utility.pickle_helper import *
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from numpy import array
import json
import codecs
import sys
import os
from nltk.metrics.distance import edit_distance
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
from nltk.corpus import wordnet
import spacy

nlp = spacy.load('en')
import threading

from concurrent.futures import *
max_workers=24

st_pos = []
st_ner = []
tag_locks = []
datas = []
for i in range(max_workers):
    st_pos.append(StanfordPOSTagger(stanford_postagger_model, path_to_jar=stanford_postagger))
    st_ner.append(StanfordNERTagger(stanford_classifier, stanford_ner_path))
    datas.append([])
    tag_locks.append(threading.Lock())



lock = threading.Lock()
def add_data(datas,snt,snt_spacy,ner,amr_t,n):
   # print (new)
    data = dict()
    data["ner"] = [n[1] for n in ner]
    data["pos"] = [w.pos_ for w in snt_spacy]
    data["amr_t"] = amr_t
    data["snt"] = snt
    data["snt_token"] = [w.text for w in snt_spacy]
    data["lemma"] = [w.lemma_ for w in snt_spacy]
    
    l = len(data["pos"])
    
    if l != len(ner):
        print ("try fix")
        print (l,len(data["ner"]),len(data["snt_token"]),len(data["lemma"]))
        data["ner"] = []
        ner_id = 0
        for t in data["snt_token"]:
            if t == ner[ner_id][0]:
                data["ner"].append(ner[ner_id][1])
                ner_id += 1
            else:
                data["ner"].append('0')
        print (ner)
        print (data["ner"])
        print (data["snt_token"])
        print (data["pos"])
        print (data["lemma"])
        print (data["snt"])
                
            
    if not ( l == len(data["ner"] ) and l == len(data["snt_token"]) and l == len(data["lemma"] )):
        print (l,len(data["ner"]),len(data["snt_token"]),len(data["lemma"]))
        print (data["pos"])
        print (data["snt_token"])
        print (data["lemma"])
        print (data["ner"])
        print (data["snt"])
        assert False
    datas[n].append(data)

NORMAL = 0    
READING_NUM = 1

def combine_num(snt_token,pos):
    new_snt_token = []
    new_pos = []
    state = NORMAL
    start_index = 0
    for i in range(len(snt_token)):
        if state == NORMAL:
            if pos[i] == "NUM":
                start_index = i
                state = READING_NUM
            else:
                new_snt_token.append(snt_token[i])
                new_pos.append(pos[i])
        else:
            if pos[i] != "NUM":
                new_snt_token.append(" ".join(snt_token[start_index:i]))
                new_pos.append("NUM")
                
                new_snt_token.append(snt_token[i])
                new_pos.append(pos[i])
                state = NORMAL
    return new_snt_token,new_pos
            
    
def generate_sentence(snt,amr_t,n,all_data):
    
    if n % 100 == 0:
        print (n)
    n = n % max_workers
    snt_spacy = nlp(snt[0:-1])
    snt_token = [w.text for w in snt_spacy]
#    snt_token,pos = combine_num(snt_token,pos)
    tag_locks[n].acquire()
    ner = [n for n in st_ner[n].tag(snt_token)]
    tag_locks[n].release()
    add_data(datas,snt,snt_spacy,ner,amr_t,n)
    
def readFile(filepath):
    for i in range(max_workers):
        datas[i] = []

    executor = ThreadPoolExecutor(max_workers)
    f = open(filepath,"r")

    line = f.readline()
    n = 0
    non_core_of_bad = 0
    core_bad = 0
    all_data = []
    while line != '' :
        if line.startswith("# ::id"):
            amr_t = ""
            n=n+1
            sid = line
            snt = f.readline().replace("# ::snt ","")
            f.readline()
            line = f.readline()
            while line.strip() != '':
                amr_t = amr_t+line
                line = f.readline()     
         #   generate_sentence(snt,amr_t,n,all_data  )
            executor.submit( generate_sentence, snt,amr_t,n,datas )
            
        #     triples = a.triples()
        #   add_embedding(snt_token,lemmas,a)
        #    add_short_list_concept(lemmas,a)
        line = f.readline()
    executor.shutdown()
    f.close()
    
    for i in range(max_workers):
        all_data += datas[i] 
    with open(filepath.replace(".txt",".json"), 'w+') as outfile:
        json.dump(all_data, outfile)
    return n


#lemmas_to_concept =  read_resource_files( f_r.get_frames())
    
    
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
    
     
print(("processing training set"))
for filepath in trainingFilesPath:
    print(("reading "+filepath.split("/")[-1]+"......"))
    n = readFile(filepath)
    print(("done reading "+filepath.split("/")[-1]+", "+str(n)+" sentences processed"))'''