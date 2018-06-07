#!/usr/bin/env python3.6
# coding=utf-8
'''

This reader reads all amr propbank file,
and add possible cannonical amr lemma
to the corresponding copying dictionary of a word and aliases of the word

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-28
'''

import xml.etree.ElementTree as ET
from nltk.stem import WordNetLemmatizer
from utility.amr import *
from utility.data_helper import folder_to_files_path
wordnet_lemmatizer = WordNetLemmatizer()

def add_concept(lemmas_to_concept,le,con):

    if not le in lemmas_to_concept:
        lemmas_to_concept[le]= set([con])
    else:
        lemmas_to_concept[le].add(con)


    
class PropbankReader:
    def parse(self):
        self.frames = dict()
        self.non_sense_frames = dict()
        self.frame_lemmas = set()
        self.joints = set()
        for f in self.frame_files_path:
            self.parse_file(f)
            
    def __init__(self, folder_path=frame_folder_path):
        self.frame_files_path = folder_to_files_path(folder_path,".xml")
        self.parse()

    def parse_file(self,f):
        tree = ET.parse(f)
        root = tree.getroot()
        for child in root:
            if child.tag == "predicate":
                self.add_lemma(child)

    #add cannonical amr lemma to possible set of words including for aliases of the words
    def add_lemma(self,node):
        lemma =  node.attrib["lemma"].replace("_","-")
        self.frames.setdefault(lemma,set())
        self.non_sense_frames.setdefault(lemma,set())
    #    self.frames[lemma] = set()
        for child in node:
            if child.tag == "roleset":
                if "." not in child.attrib["id"]:
                    if len(child.attrib["id"].split("-")) == 1:
                        le,sense = child.attrib["id"],NULL_WORD
                    else:
                        le,sense = child.attrib["id"].split("-")
                #    print (child.attrib["id"],lemma)
                else:
                    le,sense = child.attrib["id"].replace("_","-").split(".")
                self.frame_lemmas.add(le)
                role = AMRUniversal(le,Rule_Frame,"-"+sense)
                if len(role.le.split("-")) == 2:
                    k,v = role.le.split("-")
                    self.joints.add((k,v))
                no_sense_con = AMRUniversal(role.le,role.cat,None)
                add_concept(self.frames,lemma,role)
                add_concept(self.non_sense_frames,lemma,no_sense_con)
                aliases = child.find('aliases')
                if aliases:
                    for alias in aliases.findall('alias'):
                        if alias.text != le and alias.text not in self.frames:
                            alias_t = alias.text.replace("_","-")
                            add_concept(self.frames,alias_t,role)
                            add_concept(self.non_sense_frames,alias_t,no_sense_con)
        #print (le, self.frames[le])
    def get_frames(self):
        return self.frames
def main():
    f_r = PropbankReader()
    for k,v in f_r.joints:
        print (k+" "+v)


if __name__ == "__main__":
    main()
