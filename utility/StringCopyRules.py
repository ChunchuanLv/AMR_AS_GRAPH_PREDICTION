#!/usr/bin/env python
#coding=utf-8
'''
Building and hanlding category based dictionary for copying mechanism
Also used by ReCategorization to preduce training set, and templates (which partially rely on string matching).

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-28
'''

import threading
from utility.data_helper import *
from utility.AMRGraph import *
from utility.constants import *
from utility.PropbankReader import PropbankReader

from nltk.metrics.distance import edit_distance


def de_polarity(lemma):
    if len(lemma) == 0: return None
    if lemma[0] == "a" and len(lemma)> 5:
        return lemma[1:]
    if (lemma[:2]) in ["in","un","il","ir","im"] and len(lemma)>5:
        return lemma[2:]
    if lemma[:3] in ["dis","non"] and len(lemma)>6:
        return lemma[3:]
    if lemma[-4:] in ["less"] and len(lemma)>6:
        return lemma[:-4]
    return None
def polarity_match(con_lemma,lemma):
    lemma = de_polarity(lemma)
    if lemma is not None:
        if disMatch(lemma,con_lemma )<1:
            return True
    return False


#computing string dissimilarity (e.g. 0 means perfect match)
def disMatch(lemma,con_lemma,t=0.5):
 #   if (con_lemma == "and" and lemma == ";" ): return True
#    if (con_lemma == "multi-sentence" and lemma in [".",";"]): return True
    if lemma == con_lemma: return 0
    if de_polarity(lemma) == con_lemma: return 1  #not match if depolaritized matched
    if (con_lemma in lemma  or  lemma in  con_lemma and "-role" not in con_lemma) and len(lemma)>2 and len(con_lemma)>2 :
        return 0
    if lemma.endswith("ily") and lemma[:-3]+"y"==con_lemma:
        return 0
    if lemma.endswith("ing") and (lemma[:-3]+"e"==con_lemma or lemma[:-3]==con_lemma):
        return 0
    if lemma.endswith("ical") and lemma[:-4]+"y"==con_lemma:
        return 0
    if lemma.endswith("ially") and lemma[:-5] in con_lemma:
        return 0
    if lemma.endswith("ion") and (lemma[:-3]+"e"==con_lemma or lemma[:-3]==con_lemma):
        return 0
    if lemma in con_lemma and len(lemma)>3 and len(con_lemma)-len(lemma)<5:
        return 0
    if lemma.endswith("y") and lemma[:-1]+"ize"==con_lemma:
        return 0
    if lemma.endswith("er") and (lemma[:-2]==con_lemma or lemma[:-3]==con_lemma  or lemma[:-1]==con_lemma):
        return 0
    dis = 1.0*edit_distance(lemma,con_lemma)/min(12,max(len(lemma),len(con_lemma)))
    if (dis < t ) :
        return dis
    return 1

import calendar
month_abbr = {name: num for num, name in enumerate(calendar.month_abbr) if num}

_float_regexp = re.compile(r"^[-+]?(?:\b[0-9]+(?:\.[0-9]*)?|\.[0-9]+\b)(?:[eE][-+]?[0-9]+\b)?$")
def is_float_re(str):
    return re.match(_float_regexp, str)
super_scripts = '⁰¹²³⁴⁵⁶⁷⁸⁹'
def parseStr(x):
    if x.isdigit():
        if x in super_scripts:
            return super_scripts.find(x)
        return int(x)
    elif is_float_re(x):
        return float(x)
    return None

units = [
"zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
"nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
"sixteen", "seventeen", "eighteen", "nineteen",
]
tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
months = ["","january","february","march","april","may","june","july","august","september","october","november","december"]
scales = ["hundred", "thousand", "million", "billion", "trillion"]
scaless = ["hundreds", "thousands", "millions", "billions", "trillions"]
numwords = {}
numwords["and"] = (1, 0)
for idx, word in enumerate(units):  numwords[word] = (1, idx)
for idx, word in enumerate(months):  numwords[word] = (1, idx)
for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)
for idx, word in enumerate(scaless): numwords[word] = (10 ** (idx * 3 or 2), 0)
ordinal_words = {'first':1, 'second':2, 'third':3, 'fifth':5, 'eighth':8, 'ninth':9, 'twelfth':12}
ordinal_endings = [('ieth', 'y'), ('th', ''), ('st', ''), ('nd', ''), ('rd', '')]

def text2int(textnum):
    for k, v in month_abbr.items():
        if textnum == k.lower():
            return v
    if " and " in textnum :
        textnums =  textnum.split(" and ")
        out = [ j for j in [text2int(i) for i in textnums  ] if j ]
        if len(out) > 1: return sum(out)
        else: return None
    textnum = textnum.replace(',', ' ')
    textnum = textnum.replace('-', ' ')
    textnum = textnum.replace('@-@', ' ')
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






def unmixe(mixed,threshold = 50):
    high_frequency = dict()
    low_frequency = dict()
    low_text_num = dict()
    high_text_num = dict()
    for i in mixed:
        cat = i.cat
        if  mixed[i][0] >= threshold and ( cat in Rule_All_Constants) :
            high_text_num[i] = mixed[i]
        elif mixed[i][0] >= threshold:
            high_frequency[i] = mixed[i]
        elif (cat in Rule_All_Constants):
            low_text_num[i]  = mixed[i]
        else:
            low_frequency[i] = mixed[i]
        
    return high_text_num,high_frequency,low_frequency,low_text_num

class rules:
    RE_FRAME_NUM = re.compile(r'-\d\d$')
    NUM = re.compile(r'[-]?[1-9][0-9]*[:.]?[0-9]*')
    Pure_NUM = re.compile(r'[-]?[1-9][0-9]*[,]?[0-9]*')

    def save(self,filepath="data/rule_f"):
        pickle_helper= Pickle_Helper(filepath)
        pickle_helper.dump(self.lemma_back,"lemma_back")
        pickle_helper.dump([k for k in self.lemma_freq_cat.keys()],"keys")
        for cat in self.lemma_freq_cat:
            pickle_helper.dump(self.lemma_freq_cat[cat] ,cat)
        pickle_helper.save()

        self.load(filepath)

    lock = threading.Lock()
    def load(self,filepath="data/rule_f"):
        pickle_helper= Pickle_Helper(filepath)
        data = pickle_helper.load()
        keys = data["keys"]
        self.lemma_freq_cat = {}
        self.lemma_back = data["lemma_back"]
        for key in keys:
            self.lemma_freq_cat[key] = data[key]
        self.build_lemma_cheat()
        return self

    def set_rules(self):
        self.rules = {}
        self.rules[Rule_Frame]= lambda _,l,con_l ,sense: self.standard_rule(l,Rule_Frame,con_l,sense)
        self.rules[Rule_String]= lambda x,_,con_l ,__: self.standard_rule(x,Rule_String,con_l)
        self.rules[Rule_Ner]= lambda x,_ ,con_l,__:self.standard_rule(x,Rule_Ner,con_l)
        self.rules[Rule_B_Ner]= lambda x,_ ,con_l,__:self.standard_rule(x,Rule_B_Ner,con_l)
        self.rules[Rule_Constant]= lambda _,l,con_l,__: self.standard_rule(l,Rule_Constant,con_l)
        self.rules[Rule_Concept]= lambda _,l,con_l,__: self.standard_rule(l,Rule_Concept,con_l)
        self.rules[Rule_Num]= lambda _,l ,con_l,__: self.num(l)

    def entity(self,lemma,cat,con_lemma = None):
        num = self.num(lemma)
        if num is not None and num.le != NULL_WORD:
            num.cat = cat
            if cat == "date-entity" and self.Pure_NUM.search(lemma) and len(lemma) == 6:
                num.le = lemma
            return num
        return self.standard_rule(lemma,cat,con_lemma)

    def read_veb(self):
        RE_FRAME_NUM = re.compile(r'-\d\d$')
        f = open(verbalization,"r")
        f.readline()
        line = f.readline()
        while line != '' :
            tokens = line.replace("\n","").split(" ")
            if len(tokens)<= 4 and (tokens[0] =="VERBALIZE" or tokens[0] =="MAYBE-VERBALIZE"):
                old_lemma = tokens[1]
                amr_lemma = re.sub(RE_FRAME_NUM, '', tokens[3])
                if tokens[0] =="MAYBE-VERBALIZE":
                    self.add_lemma_freq(old_lemma,amr_lemma,Rule_Frame,freq=1,sense = tokens[3][-3:])
                else:
                    self.add_lemma_freq(old_lemma,amr_lemma,Rule_Frame,freq=100,sense = tokens[3][-3:])

            line = f.readline()
        f.close()


    def read_frame(self):
        f_r = PropbankReader()
        f_r = f_r.frames
        for le,concepts in f_r.items():
            i=0
            for c in concepts:
                self.add_lemma_freq(le,c.le,Rule_Frame,freq=10,sense = c.sense)
                i = i+1

    def __init__(self):
        self.lemma_freq_cat = {}
        self.lemmatize_cheat = {}
        self.lemma_back = {}
        self.read_frame()
        self.read_veb()
        self.frame_lemmas = PropbankReader().frame_lemmas
        self.build_lemma_cheat()
        self.set_rules()
     #   self.rules[Rule_Re]= lambda _,l = wordnet.NOUN: self.re(l)

    def standard_rule(self,lemma,cat,con_lemma=None,sense=NULL_WORD):
        if con_lemma is None:  #testing
            if cat in [Rule_Ner,Rule_B_Ner ]and len(lemma)>3:
                lemma = lemma.capitalize()
            if (lemma,cat) in  self.lemmatize_cheat:
            #    if lemma == "cooperation" and cat == Rule_Frame:
           #         print ("before cooperation",self.lemmatize_cheat[(lemma,cat)],AMRUniversal(lemma,cat,sense))
                lemma = self.lemmatize_cheat[(lemma,cat)]
         #       if lemma == "cooperate" and cat == Rule_Frame:
          #          print ("after cooperate",AMRUniversal(lemma,cat,sense))
         #       elif lemma == "cooperation" and cat == Rule_Frame:
          #          print ("after cooperation",AMRUniversal(lemma,cat,sense))
                return AMRUniversal(lemma,cat,sense)
            return AMRUniversal(lemma,cat,sense)
        else: #training
            if cat in [Rule_Ner,Rule_B_Ner ] and len(lemma)>3:
                lemma = lemma.capitalize()
            if cat not in self.lemma_freq_cat or lemma not in self.lemma_freq_cat[cat]:
                return AMRUniversal(lemma,cat,sense)
            candidates = self.lemma_freq_cat[cat][lemma]
            if con_lemma in candidates.keys():
                return AMRUniversal(con_lemma,cat,sense)
            return AMRUniversal(lemma,cat,sense)

    def clear_freq(self):
        self.lemma_freq_cat = {}
        self.lemmatize_cheat = {}

    def add_lemma_freq(self,old_lemma,amr_lemma,cat,freq=1,sense=NULL_WORD):
     #   if cat in Rule_All_Constants:
     #       return
        self.lock.acquire()
        if old_lemma == amr_lemma: freq *= 10
        amr_con = amr_lemma
        self.lemma_back[amr_con][old_lemma] = self.lemma_back.setdefault(amr_con,{}).setdefault(old_lemma,0)+freq
        lemma_freq = self.lemma_freq_cat.setdefault(cat,{}).setdefault(old_lemma,{})
        lemma_freq[amr_con] = lemma_freq.setdefault(amr_con,0)+freq
        self.lock.release()
            
        
    def build_lemma_cheat(self):
        for cat in self.lemma_freq_cat:
            lemma_freqs = self.lemma_freq_cat[cat]
            for word in lemma_freqs:
                max_score = 0
                max_lemma = word
                for arm_le in lemma_freqs[word]:
                    score =  1.0*lemma_freqs[word][arm_le]
                    assert (score > 0)
                    if score >max_score:
                        max_score = score
                        max_lemma = arm_le
                self.lemmatize_cheat[(word,cat)] = max_lemma

    #    print (self.lemmatize_cheat)
            
   # fragments_to_break = set(["up","down","make"])
    
    def num(self,lemma):
        r = text2int(lemma)
        if r is None and self.Pure_NUM.search(lemma) is not None:
            lemma = lemma.replace(",","")
            return AMRUniversal(lemma,Rule_Num,None)
        if r is not None:
            return AMRUniversal(str(r),Rule_Num,None)
        return AMRUniversal(NULL_WORD,Rule_Num,None)

    #old_ids : batch x (cat,le,lemma,word) only cat is id
    def toAmrSeq(self,cats,snt,lemma,high,auxs,senses = None,ners = None):
        out = []
        for i in range(len(snt)):
            sense  = senses[i] if senses else None
            txt, le,cat,h,aux  = snt[i],lemma[i],cats[i],high[i],auxs[i]
            assert h is None or isinstance(h,str) or isinstance(h,tuple)and isinstance(cat,str) ,(txt, le,cat,h)
            if h and h != UNK_WORD:
                if cat == Rule_Num:
                    uni =  self.to_concept(txt,h,Rule_Num,sense)
                    if uni.le == NULL_WORD:
                        uni =  AMRUniversal(h,Rule_Concept,sense)
                else:
                    uni = AMRUniversal(h,cat,sense)
            else:
                try_num = self.to_concept(txt,le,Rule_Num,sense)
                if " " in le and try_num.le != NULL_WORD and cat not in [Rule_String,Rule_B_Ner,Rule_Ner] and "entity" not in cat:
                    uni = try_num
                else:
                    uni = self.to_concept(txt,le,cat,sense)

            if cat == Rule_B_Ner:
                if not aux in [UNK_WORD,NULL_WORD]:
                    uni.aux = aux
                elif ners[i] == "PERSON":
                    uni.aux = "person"
                elif ners[i] == "LOCATION":
                    uni.aux = "location"
                elif ners[i] == "ORGANIZATION":
                    uni.aux = "organization"
                else:
                    uni.aux = UNK_WORD
            assert  isinstance(uni.le,str) and isinstance(uni.cat,str ),(txt, le,cat,h,uni.le,uni,cat)


            if ners[i] == "URL": #over write ML decision, otherwise creating bug
                uni =  AMRUniversal(le,"url-entity",None)

            out.append(uni)

        return out
    

    def to_concept(self,txt,le,cat,con_lemma=None,sense=NULL_WORD):
        if cat in self.rules:
            return self.rules[cat](txt,le,con_lemma,sense)
        elif cat.endswith("-entity"):   # entity
            return self.entity(le,cat,con_lemma)
        else:
            return self.standard_rule(le,cat,con_lemma)


    #amr is myamr
    def get_matched_concepts(self,snt_token,amr_or_node_value,lemma,pos,with_target = False,jamr=False,full=1):
        results = []
        node_value = amr_or_node_value.node_value(keys=["value","align"]) if isinstance(amr_or_node_value,AMRGraph) else amr_or_node_value
        for n,c,a in node_value:
            if full == 1:
                align = self.match_concept(snt_token,c,lemma,pos,with_target)
                if jamr and a is not None:
                    exist = False
                    for i,l,p in align:
                        if i == a:
                            exist = True
                    if not exist:
                        align += [(a,lemma[a],pos[a])]
                results.append([n,c,align])
            else:
                if jamr and a is not None:
                    align = [(a,lemma[a],pos[a])]
                else:
                    align = self.match_concept(snt_token,c,lemma,pos,with_target)
                results.append([n,c,align])
        return results

    def match_concept(self,snt_token,concept,lemma,pos,with_target = False,candidate = None):
        if len(lemma) == 1: return [[0,lemma[0],pos[0]]]
        le,cat,sense = decompose(concept)
        align = []
        if candidate is None:
            candidate = range(len(snt_token))
        for i in candidate:
            if with_target and  disMatch(lemma[i],le) <1: # and pos[i] not in ["IN"]:
                align.append((i,lemma[i],pos[i]))
                continue
            if with_target:
                amr_c = self.to_concept(snt_token[i],lemma[i],cat,le,sense)
            else:
                amr_c = self.to_concept(snt_token[i],lemma[i],cat)
            if amr_c is None:
                continue
            le_i,cat_i,sen_i = decompose(amr_c)

            assert cat == cat_i, "cat mismatch "+ snt_token[i]+" "+lemma[i]+" "+cat+" "+le+" "+cat_i+" "+le_i+"\n"+" ".join(snt_token)
            if amr_c.non_sense_equal(concept): #  and pos[i] not in ["IN"]:
                align.append((i,lemma[i],pos[i]))

        if le == "and" and len(align) == 0:
            for i in range(len(lemma)):
                if lemma[i] == ";" or lemma[i] == "and":
                    align.append((i,lemma[i],pos[i]))
            if len(align)>0: return [align[-1]]

         #   if len(align) > 0 : print (le,align,lemma)

        if le == "multi-sentence" and len(align) == 0 and False:
            for i in range(len(lemma)):
                if lemma[i] in [".",";","?","!"]:
                    align.append((i,lemma[i],pos[i]))
                    return align
        #    if len(align) > 0 : print (le,align,lemma)
        return align

