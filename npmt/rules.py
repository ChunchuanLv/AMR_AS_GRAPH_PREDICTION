#!/usr/bin/env python
#coding=utf-8

import threading
import operator
from utility.pickle_helper import *
from concurrent.futures import *
import pickle
from utility.constants import *
from utility.converter import *
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
import re,string
from utility.amr import *
from utility.__init__ import *
from utility.constants import *
from utility.reader import Coarse_Frames_Reader
from nltk.metrics.distance import edit_distance
#from nltk.tag import StanfordPOSTagger
#st_pos = StanfordPOSTagger(stanford_postagger_model, path_to_jar=stanford_postagger)
from nltk.corpus import wordnet
import calendar
month_abbr = {name: num for num, name in enumerate(calendar.month_abbr) if num}


_float_regexp = re.compile(r"^[-+]?(?:\b[0-9]+(?:\.[0-9]*)?|\.[0-9]+\b)(?:[eE][-+]?[0-9]+\b)?$")
def is_float_re(str):
    return re.match(_float_regexp, str)

parseStr = lambda x:  x.isdigit() and int(x)  or is_float_re(x) and float(x) or None

def text2int(textnum, numwords={}):
    if not numwords:
        units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
        ]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        months = ["","January","February","March","April","May","June","July","August","September","October","November","December"]
        scales = ["hundred", "thousand", "million", "billion", "trillion"]
        scaless = ["hundreds", "thousands", "millions", "billions", "trillions"]
        month = "Jun"
        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):  numwords[word] = (1, idx)
        for idx, word in enumerate(months):  numwords[word] = (1, idx)
        for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)
        for idx, word in enumerate(scaless): numwords[word] = (10 ** (idx * 3 or 2), 0)

    ordinal_words = {'first':1, 'second':2, 'third':3, 'fifth':5, 'eighth':8, 'ninth':9, 'twelfth':12}
    ordinal_endings = [('ieth', 'y'), ('th', ''), ('st', ''), ('nd', ''), ('rd', '')]
    for k, v in month_abbr.items():
        if textnum == k.lower():
            return v
    textnum = textnum.replace('-', ' ')
    current = result = 0
    for word in textnum.split():
        w_num = parseStr(word)
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
        elif w_num:
            scale, increment = (1,w_num)
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = "%s%s" % (word[:-len(ending)], replacement)
            if word not in numwords:
                return None
            scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0
    return int(result + current)

def read_mor(lemmas_to_concept):
    NOUN = re.compile('::DERIV-NOUN ')
    VERB = re.compile('::DERIV-VERB ')
    ACTOR = re.compile("::DERIV-NOUN-ACTOR")
    def link_remaining(tokens):
        le = tokens[1]
     #   print (le)
        if le in  lemmas_to_concept:
            concept = lemmas_to_concept[le]
            for i in range(3,len(tokens),2):
                lemmas_to_concept[tokens[i]]= concept
              #  print (tokens[i],concepts)
    f = open(morph_verbalization,"r")
    f.readline()
    line = f.readline()
    while line != '' :
        noun = re.search(NOUN, line)
        verb = re.search(VERB, line)
        tokens = line.replace("\"","").split()
        link_remaining(tokens)
   #     print (tokens)
      #  print(verb,noun)
            #add_concept(lemmas_to_concept,tokens[1],name)
        line = f.readline()
    f.close()
    
def read_veb(frame_list):
    RE_FRAME_NUM = re.compile(r'-\d\d$')
    def add_remaining(tokens):
        le = tokens[1]
        distance = dict()
        for i in range(3,len(tokens),3):
            if RE_FRAME_NUM.search(tokens[i]) is not None :
                s = re.sub(RE_FRAME_NUM, '', tokens[i]) 
                distance[s] = edit_distance(le,s)
        mini = 99999999
        r = ""
        for s,dis in distance.items():
            if dis <mini:
                mini = dis
                r = s
        if mini != 99999999:
            frame_list[le] = Concept(r+"-99")

    f = open(verbalization,"r")
    f.readline()
    line = f.readline()
    while line != '' :
        tokens = word_tokenize(line)
        if tokens[0] =="VERBALIZE" or tokens[0] =="MAYBE-VERBALIZE":
            add_remaining(tokens)
            #add_concept(lemmas_to_concept,tokens[1],name)
        line = f.readline()
    f.close()
def read_frame(le_to_concept):
    RE_FRAME_NUM = re.compile(r'-\d\d$')
    f_r = Coarse_Frames_Reader().frames
    for le,concepts in f_r.items():
        i=0
        for c in concepts:
            t =  re.sub(RE_FRAME_NUM,"",c._name)
            if (i>=1) and t != re.sub(RE_FRAME_NUM,"",le_to_concept[le]._name):
                print (le,concepts)
                break
            le_to_concept[le] = Concept(t+"-99")
            i = i+1


#or (s1 in s2 and len(s1) > 4)or (s2 in s1 and len(s2) > 4)
def checkMatch(lemma,con_lemma,matches = [0,0,0,0,0,0,0],original = True):

    if con_lemma in lemma  and len( con_lemma)>2 and len(lemma)-len(con_lemma)<5:  
        matches[0] += 1
        return True
    if lemma.endswith("ily") and lemma[:-3]+"y"==con_lemma: 
        matches[1] += 1
        return True
    if lemma.endswith("ing") and lemma[:-3]+"e"==con_lemma:  
        matches[2] += 1
        return True
    if lemma.endswith("ical") and lemma[:-4]+"y"==con_lemma: 
        matches[3] += 1
        return True
    if (1.0*edit_distance(lemma,con_lemma)/min(len(lemma),len(con_lemma)) < 0.3) :
        matches[4] += 1
        return True
    if lemma.endswith("ion") and lemma[:-3]+"e"==con_lemma: 
        matches[5] += 1
        return True
    if lemma in con_lemma and len(lemma)>3 and len(con_lemma)-len(lemma)<5:  
        matches[6] += 1
        return True
    if lemma.endswith("y") and lemma[:-1]+"ize"==con_lemma:  
        matches[7] += 1
        return True
    if (original and lemma.startswith("in") or  lemma.startswith("un")) and len(lemma)>5:
        lemma = lemma[2:]
        return checkMatch(lemma,con_lemma,matches,False )
    return False


def unmixe(mixed,threshold = 50):
    high_frequency = dict()
    low_frequency = dict()
    text_num = dict()
    for i in mixed:
        if  mixed[i][0] > threshold and i.is_constant() and (i.is_str() or i.is_num()):
            text_num[i] = mixed[i]
        elif mixed[i][0] > threshold:
            high_frequency[i] = mixed[i]
        else:
            low_frequency[i] = mixed[i]
        
    return text_num,high_frequency,low_frequency


def save_rule(rl,filepath):
    pickle_helper= Pickle_Helper(filepath) 
    pickle_helper.dump(rl.frame_list,"frame_list")
    pickle_helper.dump(rl.broken_concepts,"broken_concepts")
    pickle_helper.dump(rl.lemma_freq_con,"lemma_freq_con")
    pickle_helper.dump(rl.lemma_freq_frame,"lemma_freq_frame")
    pickle_helper.dump(rl,"rl")
    pickle_helper.save()
    
    
def load_rule(filepath):
    pickle_helper= Pickle_Helper(filepath) 
    data = pickle_helper.load()
    rl = data["rl"]
    rl.frame_list = data["frame_list"]
    rl.broken_concepts = data["broken_concepts"]
    rl.lemma_freq_con = data["lemma_freq_con"]
    rl.lemma_freq_frame = data["lemma_freq_frame"]
    rl.build_lemma_cheat()
    rl.rules[Rule_Frame]= lambda _,l = wordnet.VERB: rl.frame99(l)
    rl.rules[Rule_String]= lambda x,_ = wordnet.NOUN: rl.arm_string(x) 
    rl.rules[Rule_Constant]= lambda x,_ = wordnet.NOUN: rl.arm_constant(x) 
    rl.rules[Rule_Concept]= lambda _,l = wordnet.VERB: rl.concept(l) 
    rl.rules[Rule_Num]= lambda _,l = wordnet.NOUN: rl.num(l) 
    return rl

class rules:
    frame_list = dict()
    lemmatize_cheat_frame = dict()
    lemma_freq_frame = dict()
    lemmatize_cheat_con = dict()
    lemma_freq_con = dict()
    broken_concepts = dict()
    rules = dict()
    RE_FRAME_NUM = re.compile(r'-\d\d$')
    NUM = re.compile(r'[-]?[1-9][0-9]*[:.]?[0-9]*')
    Pure_NUM = re.compile(r'[-]?[1-9][0-9]*[,]?[0-9]*')
    
    lock = threading.Lock()
    def __init__(self):
        read_veb(self.frame_list)
        read_frame(self.frame_list)
        read_mor(self.frame_list)
        self.rules[Rule_Frame]= lambda _,l = wordnet.VERB: self.frame99(l)
        self.rules[Rule_String]= lambda x,_ = wordnet.NOUN: self.arm_string(x) 
        self.rules[Rule_Constant]= lambda _,l = wordnet.NOUN: self.arm_constant(l) 
        self.rules[Rule_Concept]= lambda _,l = wordnet.VERB: self.concept(l) 
        self.rules[Rule_Num]= lambda _,l = wordnet.NOUN: self.num(l) 
    
     #   for k,d in self.verbalization_list.items():
     #       print (k,d)
    
    def add_lemma_freq(self,old_lemma,lemma,is_frame):
        if is_frame:
            lemma_freq = self.lemma_freq_frame
        else:
            lemma_freq = self.lemma_freq_con
        
        self.lock.acquire()
        if old_lemma not in lemma_freq:
            lemma_freq[old_lemma] = dict()
            lemma_freq[old_lemma][old_lemma] = 1
            lemma_freq[old_lemma][lemma] = 1
        elif lemma not in lemma_freq[old_lemma]:
            lemma_freq[old_lemma][lemma] = 1
        else:
            lemma_freq[old_lemma][lemma] += 1
        self.lock.release()
            
        
    def build_lemma_cheat(self):
        lemma_freq = self.lemma_freq_con
        lemmatize_cheat = self.lemmatize_cheat_con
        for word in lemma_freq:
            freqs = lemma_freq[word]
            max_lemma = max(freqs, key=freqs.get)
            lemmatize_cheat[word] = max_lemma
                
        lemma_freq = self.lemma_freq_frame
        lemmatize_cheat = self.lemmatize_cheat_frame
        for word in lemma_freq:
            freqs = lemma_freq[word]
            max_lemma = max(freqs, key=freqs.get)
            lemmatize_cheat[word] = max_lemma
            
   # fragments_to_break = set(["up","down","make"])
    
    def add_broken_concepts(self,composite):
        composite_t = re.sub(composite.RE_FRAME_NUM,"",composite.__str__())
        if not ("-" in composite_t) or composite in self.broken_concepts:
            return False
        c_ts = composite_t.split("-")
        
        self.lock.acquire()
        self.broken_concepts[composite] = tuple([Concept(t) for t in c_ts])
        self.broken_concepts[self.broken_concepts[composite]] = composite
        self.lock.release()
        return True
        
    def arm_string(self,s):
        return AMRString(s)
    
    def arm_constant(self,lemma):
        return AMRConstant(lemma)
    
    def concept(self,lemma):
        if lemma in self.lemmatize_cheat_con:
            return Concept(self.lemmatize_cheat_con[lemma])
        else:
            return Concept(lemma)
    
    def num(self,lemma):
        if self.Pure_NUM.search(lemma) is not None :
            return AMRNumber(lemma.replace(",",""))
        r = text2int(lemma)
        if r:
            return AMRNumber(str(r))
    
        
    def frame99(self, lemma):
        #can insert exceptions here
        if lemma in self.lemmatize_cheat_frame:
            lemma = self.lemmatize_cheat_frame[lemma]
        if lemma in self.frame_list:
            return self.frame_list[lemma]
        return Concept(lemma+"-99")


    def apply_all_sentence(self,snt_token,lemma):
        results = dict()
        for i in range(len(snt_token)):
            for n in self.rules:
                r = self.rules[n]
                k = r(snt_token[i],lemma[i])
                if k:
                    if k in results:
                        if not ((n,lemma[i],i) in results[k]):
                         #   print (k,results[k] , (n,lemma[i]))
                            results[k].append((n,lemma[i],i,k))
                            for i in range(len(results[k])):
                                if n != results[k][i][0]:
                                    print (k,results[k] )
                    else:
                        results[k] = [(n,lemma[i],i,k)]
        return results
    
    def convert_concept(self,con):
        if (con in self.broken_concepts and False ):
            return self.broken_concepts[con]
        else:
            if con.is_frame():
                return Concept(re.sub(con.RE_FRAME_NUM,"-99",con.__str__()))
            else:
                return con
            
    def get_matched_concepts(self,snt_token,amr,lemma):
        targets = []
        for h, r, d in amr.triples():
            if d.is_var():
                targets.append(self.convert_concept(amr._v2c[d]))
            elif d.is_constant() and r != ":wiki":
                targets.append(d)
                
        #    targets.append(con[1])
    #    targets = targets+ [c for c in amr.constants()]
        rule_produced_indexed = self.apply_all_sentence(snt_token,lemma)
        for con in targets:
            if con in rule_produced_indexed:
                for n,content,index in  rule_produced_indexed[con]:
                    if categorize(con) == n:
                        self.add_lemma_freq(content,re.sub(self.RE_FRAME_NUM,"",con.__str__()), n == Rule_Frame)
                    
    def get_none_rule(self,snt_token,amr,lemma):
        targets = []
        for h, r, d in amr.triples():
            if d.is_var():
                targets.append(self.convert_concept(amr._v2c[d]))
            elif d.is_constant() and r != ":wiki":
                targets.append(d)
                
        rule_produced_indexed = self.apply_all_sentence(snt_token,lemma)
        none = []
        for t in targets:
            if not t in rule_produced_indexed:
                none.append(t)
        return none
def main():
    
    import spacy
    nlp = spacy.load('en') 
    
    rl = rules()
    a = AMR("(v / visit-01 :ARG0 (w / we) :ord (o / ordinal-entity :value 1 :range (t / temporal-quantity :quant 50000 :unit (y / year))))")
    s = "our first visit in 10 years 50,000"
    x = nlp(s)
    lemma = [i.lemma_ for i in x]
    snt_token = [i.text for i in x]
    print (rl.get_none_rule(snt_token,a,lemma))
   # print (text2int("one hundred and ninety nine"))


if __name__ == "__main__":
    main()
