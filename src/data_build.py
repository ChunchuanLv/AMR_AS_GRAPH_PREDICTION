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
    dictionary.add(EOS_WORD)
            
def seq_to_id(dictionary,seq):
    id_seq = []
    freq_seq = []
    for i in seq:
        id_seq.append(dictionary[i])
        freq_seq.append(dictionary.frequencies[dictionary[i]])
    id_seq.append(EOS)
    freq_seq.append(dictionary.frequencies[EOS])
    return id_seq,freq_seq
            
#id_seq :  [(con_lemma,cat,rel])]
def amr_seq_to_id(lemma_dict,concept_dict,category_dict,rel_dict,amr_seq):
    id_seq = [[BOS,BOS,BOS]]
    for l in amr_seq:
        id_seq.append([lemma_dict[l[3]],category_dict[l[1]],rel_dict[l[2]]]) 
    id_seq.append([EOS,EOS,EOS])
    return id_seq

#id_seq :  [(con_lemma,cat,[id],rel])]
def amr_seq_to_index(amr_seq,snt_len):
    src_index_seq = [[-2]] #just a dumb
    re_entrance_seq = [0]
    for l in amr_seq:
        src_index_seq.append([t[2] for t in l[0]]) 
        re_entrance_seq.append(l[-1]+1)
    src_index_seq.append([snt_len-1])
    re_entrance_seq.append(len(re_entrance_seq))
    return src_index_seq,re_entrance_seq

  
def amr_seq_to_dict(lemma_dict,concept_dict,category_dict,rel_dict,seq):
    for i in seq:
        category_dict.add(i[1])      
        rel_dict.add(i[2])       
        lemma_dict.add(i[3])
        for t in i[0]:
            if t[1] == HIGH:
                concept_dict.add(i[3],lemma_dict[i[3]])   #concept_dict in lemma_dict
def handle_sentence(data,filepath,build_dict,n):
    
    if n % 1000 == 0:
        print (n)

    ner = data["ner"]
    snt_token = data["snt_token"]
    pos = data["pos"]
    lemma =  data["lemma"]
    
        
    amr_t = data["amr_t"]
    amr = AMR(amr_t)
    amr_seq = amr_to_seq(amr,snt_token,lemma,rl,high_freq)
    data["amr_seq"] = amr_seq      #[[[lemma1],category,relation]]
    
    if build_dict:
        add_seq_to_dict(word_dict,snt_token)
        add_seq_to_dict(lemma_dict,lemma)
        add_seq_to_dict(pos_dict,pos)
        add_seq_to_dict(ner_dict,ner)         
        amr_seq_to_dict(lemma_dict,concept_dict,category_dict,rel_dict,data["amr_seq"])
    else:
        data["snt_id"],data["src_freq"] = seq_to_id(word_dict,snt_token)
        data["lemma_id"] = seq_to_id(lemma_dict,lemma)[0]
        data["pos_id"] = seq_to_id(pos_dict,pos)[0]
        data["ner_id"] = seq_to_id(ner_dict,ner)[0]
        l = len(data["pos_id"])
        if not ( l == len(data["snt_id"]) and l == len(data["lemma_id"]) and l == len(data["ner_id"])):
            print (l,len(data["snt_id"]),len(data["lemma_id"]),len(data["ner_id"]))
            print (data["pos_id"])
            print (data["snt_id"])
            print (data["lemma_id"])
            print (data["ner_id"])
            print (pos)
            print (snt_token)
            print (lemma)
            print (ner)
            print (data["snt"])
            assert(False)
        data["amr_id"] = amr_seq_to_id(lemma_dict,concept_dict,category_dict,rel_dict,amr_seq)
        data["index"] = amr_seq_to_index(amr_seq,l)

    
def readFile(filepath,build_dict = False):

    
    with open(filepath.replace(".txt",".json"),'r') as data_file:    
        all_data = json.load(data_file)
    n = 0
    for data in all_data:
        n = n+1
        handle_sentence(data,filepath,build_dict,n)
    if not build_dict:
        outfile = Pickle_Helper(filepath.replace(".txt",".pickle")) 
        outfile.dump(all_data, "data")
        outfile.save()
    return len(all_data)

rl = load_rule("data/rule_f")

non_rule_set_f = Pickle_Helper("data/non_rule_set") 
non_rule_set = non_rule_set_f.load()["non_rule_set"]
threshold = 5
high_text_num,high_non_text,low_non_text=unmixe(non_rule_set,threshold)
high_freq = dict(list(high_non_text.items()) + list(high_text_num.items()))
print(len(high_non_text),len(high_text_num),len(high_freq))



def initial_dict(filename):
    d = Dict(filename)
    d.addSpecial(PAD_WORD,PAD)
    d.addSpecial(UNK_WORD,UNK)
    d.addSpecial(BOS_WORD,BOS)
    d.addSpecial(EOS_WORD, EOS)
    return d

word_dict = initial_dict("data/word_dict")
lemma_dict = initial_dict("data/lemma_dict")
pos_dict = initial_dict("data/pos_dict")
ner_dict = initial_dict("data/ner_dict")
concept_dict = initial_dict("data/concept_dict")

rel_dict = initial_dict("data/rel_dict")
category_dict = initial_dict("data/category_dict")

for filepath in trainingFilesPath:
    print(("reading "+filepath.split("/")[-1]+"......"))
    n = readFile(filepath,build_dict = True)
    print(("done reading "+filepath.split("/")[-1]+", "+str(n)+" sentences processed"))


word_dict.save()
lemma_dict.save()
pos_dict.save()
ner_dict.save()
rel_dict.save()
concept_dict.save()
category_dict.save()

print("dictionary building done")
print("word_dict lemma_dict pos_dict ner_dict concept_dict rel_dict")
print(len(word_dict),len( lemma_dict),len( pos_dict),len( ner_dict ),len(concept_dict),len(rel_dict))



high_freq_no_99 = [le.__str__().replace("-99","") for le in high_freq.keys()]

diff = []
for le in high_freq_no_99:
    if le not in concept_dict:
        diff.append(le)
print(diff)
print(("processing training set"))
for filepath in trainingFilesPath:
    print(("reading "+filepath.split("/")[-1]+"......"))
    n = readFile(filepath,build_dict = False)
    print(("done reading "+filepath.split("/")[-1]+", "+str(n)+" sentences processed"))
   
    
print(("processing development set"))
for filepath in devFilesPath:
    print(("reading "+filepath.split("/")[-1]+"......"))
    n = readFile(filepath,build_dict = False)
    print(("done reading "+filepath.split("/")[-1]+", "+str(n)+" sentences processed"))
    
    
print(("processing test set"))
for filepath in testFilesPath:
    print(("reading "+filepath.split("/")[-1]+"......"))
    n = readFile(filepath,build_dict = False)
    print(("done reading "+filepath.split("/")[-1]+", "+str(n)+" sentences processed"))
     
    
print(len(high_non_text),len(high_text_num),len(high_freq))
