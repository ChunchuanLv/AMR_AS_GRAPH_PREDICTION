import pickle
from utility.constants import *
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from utility.amr import *
wordnet_lemmatizer = WordNetLemmatizer()

def add_concept(lemmas_to_concept,le,co):
    if not le in lemmas_to_concept:
        lemmas_to_concept[le] = set([co])  
    else:
        lemmas_to_concept[le].add(co)
    
def read_91(lemmas_to_concept,path,name):
    f = open(path,"r")
    f.readline()
    role = Concept(name)
    line = f.readline()
    while line != '' :
        tokens = word_tokenize(line)
        add_concept(lemmas_to_concept,tokens[1],role)
        line = f.readline()

def read_veb(lemmas_to_concept):
    def add_remaining(tokens):
        le = tokens[1]
        for i in range(3,len(tokens),3):
            add_concept(lemmas_to_concept,le,Concept(tokens[i]))
    f = open(verbalization,"r")
    f.readline()
    line = f.readline()
    while line != '' :
        tokens = word_tokenize(line)
        if tokens[0] =="VERBALIZE" or tokens[0] =="MAYBE-VERBALIZE":
            if wordnet_lemmatizer.lemmatize(tokens[1],'v') != tokens[1]:
      #          print (wordnet_lemmatizer.lemmatize(tokens[1],'v'),tokens[1:])
                add_remaining(tokens)
            #add_concept(lemmas_to_concept,tokens[1],name)
        line = f.readline()
    f.close()
def read_mor(lemmas_to_concept):
    NOUN = re.compile('::DERIV-NOUN ')
    VERB = re.compile('::DERIV-VERB ')
    ACTOR = re.compile("::DERIV-NOUN-ACTOR")
    def link_remaining(tokens):
        le = tokens[1]
     #   print (le)
        if le in  lemmas_to_concept:
            concepts = lemmas_to_concept[le]
            for i in range(3,len(tokens),2):
                if tokens[i] in lemmas_to_concept:
                    lemmas_to_concept[tokens[i]].update(concepts)
                else:
                    lemmas_to_concept[tokens[i]]= concepts
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
    
class Coarse_Frames_Reader:
    def parse(self):
        self.frames = dict()
        for f in self.frame_files_path:
            self.parse_file(f)
            
    def __init__(self, frame_folder_path):
        self.frame_files_path = folder_to_files_path(frame_folder_path,".xml")
        self.parse()
            
    def __init__(self):
        self.frame_files_path = frame_files_path
        self.parse()



    def parse_file(self,f):
        tree = ET.parse(f)
        root = tree.getroot()
        for child in root:
            if child.tag == "predicate":
                self.add_lemma(child)
                
    def add_lemma(self,node):
        le =  node.attrib["lemma"].replace("_","-")
        role = Concept(le+"-99")
        self.frames[le] = set([role]) 
        for child in node:
            if child.tag == "roleset":
                aliases = child.find('aliases')
                if aliases:
                    for alias in aliases.findall('alias'):
                        if alias.text != le and alias.text not in self.frames:
                            add_concept(self.frames,alias.text,role)
        #print (le, self.frames[le])
    def get_frames(self):
        return self.frames
def main():
    f_r = Coarse_Frames_Reader()
    print (len(f_r.frames))


if __name__ == "__main__":
    main()
