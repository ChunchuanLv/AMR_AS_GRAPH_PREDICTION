#!/usr/bin/env python2.7
# coding=utf-8
from utility.amr import *
from npmt.rules import *
from utility.constants import *

        
def categorize(c):
    if c.is_constant():
        if c.is_str():
            return Rule_String
        if c.is_num():
            return Rule_Num
        return Rule_Constant
    if c.is_frame():
        return Rule_Frame
    return Rule_Concept

def tuple_list_convert(tuple_list,cat,amr,snt_token):
    converted = []
    for n,lemma,i,k in tuple_list:
        if cat == n:
            converted.append((lemma,RULE,i))
        else:
            print ((cat,n,lemma,RULE,i))
    if (len(converted)==0):
        print (tuple_list,cat,amr,snt_token)
    return converted
import math
def sumdistance(concept_index,word_index,output_concepts,snt_len):
    distance = 0
    range_to_compare = 1
    while distance <= 0.0001 and range_to_compare<30:
        for d in range(len(output_concepts)): 
            if d != concept_index and range_to_compare-1<=abs(d-concept_index)<range_to_compare:
                for tup in output_concepts[d][0]:
                    if tup[2] != -1:
                        distance += 1.0*abs(tup[2]-word_index)* 1.0/math.exp(-2*abs(d-concept_index))
        range_to_compare +=1
    return distance

def chooseOne(i,output_concepts,snt_len):
    distances = [None]*len(output_concepts[i][0])
    for d in range(len(distances)):
        distances[d] =  sumdistance(i, output_concepts[i][0][d][2],output_concepts,snt_len)
    
    min_index, min_value = min(enumerate(distances), key=operator.itemgetter(1))
    print (distances,output_concepts[i],i)
    return ([output_concepts[i][0][min_index]],output_concepts[i][1],output_concepts[i][2])

def amr_to_seq(amr,snt_token,lemma,rl,high_freq):   #high_freq should be a dict()
    rel_concepts = amr.dfs_index_re()
    rule_produced = rl.apply_all_sentence(snt_token,lemma)
    output_concepts = []
    used_index = []
    i = 0
    for rel_con in rel_concepts:
        r = rel_con[0]
        c = rel_con[1]
        first_i = rel_con[2]
        cat = categorize(c)
        if cat == Rule_Frame:
            c = Concept(re.sub(RE_FRAME_NUM,"-99",c.__str__()))
        else:
            c = c
        if r != ":wiki":
            lemma_list = []
            if (c in rule_produced ):
                lemma_list = tuple_list_convert(rule_produced[c],cat,amr,snt_token)  #[(lemma,rule,index)]
            if (c in high_freq):
                if cat == Rule_Frame:
                    lemma_list.append((re.sub(RE_FRAME_NUM,"",c.__str__()),HIGH,-1))
                else:
                    lemma_list.append((c.__str__(),HIGH,-1))
            if (len(lemma_list)==0) :
                lemma_list = [(UNK_WORD,LOW,-2)]
            if cat == Rule_Frame:
                output_concepts.append( (lemma_list,  cat,  r,  re.sub(RE_FRAME_NUM,"",c.__str__()),first_i))
            else:
                output_concepts.append( (lemma_list,  cat,  r,  c.__str__(),first_i))
            i += 1
          #  if (output_concepts[-1]==[]):
         #       output_concepts[-1].append((categorize(c),[]))
    return output_concepts   #[[[lemma1,lemma2],category,relation]]

def fake_amr_to_seq(amr,snt_token,lemma,rl,high_freq):   #high_freq should be a dict()
    rel_concepts = amr.bfs()
    rule_produced = rl.apply_all_sentence(snt_token,lemma)
    output_concepts = []
    used_index = []
    i = 0
    for rel_con in rel_concepts:
        r = rel_con[0]
        c = rel_con[1]
        cat = categorize(c)
        if cat == Rule_Frame:
            c = Concept(re.sub(RE_FRAME_NUM,"-99",c.__str__()))
        else:
            c = c
        if r != ":wiki":
            lemma_list = []
            if (c in rule_produced ):
                lemma_list = tuple_list_convert(rule_produced[c],cat,amr,snt_token)  #[(lemma,rule,index)]
            if (c in high_freq):
                if cat == Rule_Frame:
                    lemma_list.append((re.sub(RE_FRAME_NUM,"",c.__str__()),HIGH,-1))
                else:
                    lemma_list.append((c.__str__(),HIGH,-1))
            if (len(lemma_list)==0) :
                lemma_list = [(UNK_WORD,LOW,-2)]
            if cat == Rule_Frame:
                output_concepts.append( (lemma_list,  cat,  r,  re.sub(RE_FRAME_NUM,"",c.__str__())))
            else:
                output_concepts.append( (lemma_list,  cat,  r,  c.__str__()))
            i += 1
          #  if (output_concepts[-1]==[]):
         #       output_concepts[-1].append((categorize(c),[]))
    return output_concepts   #[[[
def extract_ne_v(amr,v_e,v_n):
    wiki = AMRNumber("-")
    names = []
    for v,r,c in amr.role_triples():
        if r == ":wiki" and v == v_e:
            wiki = c
        if r.startswith(":op") and v == v_n:
            names.append((int(r[3]),c))
    names = tuple([c[1] for c in sorted(names, key=lambda i:i[0])])
    n_to_g = dict()
    n_to_g[names] = dict()
    n_to_g[names][":wiki"] = wiki
    for i in amr.concepts():
        if i[0] == v_e:
           n_to_g[names]["entity"] = i[1]
           return  n_to_g
                  
def extract_ne(amr):
    named_entities = []
    for v_e,r,v_n in amr.role_triples():
        if r == ":name":
            named_entities.append(extract_ne_v(amr,v_e,v_n))
    return named_entities
            
def test():
    amr_t = "(b / byline-91 :ARG0 (p2 / publication :wiki \"Xinhua_News_Agency\" :name (n / name :op1 \"Xinhua\" :op2 \"News\" :op3 \"Agency\")) :ARG1 (p / person :wiki - :name (n2 / name :op1 \"Yongfeng\" :op2 \"Shi\") :ARG0-of (r / report-01)) :time (d / date-entity :month 3 :day 8) :location (c2 / city :wiki \"Yichang\" :name (n3 / name :op1 \"Yichang\")))"
    snt = "Xinhua News Agency , Yichang , March 8th , by reporter Yongfeng Shi"

    amr = AMR(amr_t)    
    print (extract_ne(amr))
    return None

if __name__ == '__main__':
    test()
