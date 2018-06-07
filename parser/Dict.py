from utility.amr import *
from utility.data_helper import *
import torch

def seq_to_id(dictionary,seq):
    id_seq = []
    freq_seq = []
    for i in seq:
        id_seq.append(dictionary[i])
        freq_seq.append(dictionary.frequencies[dictionary[i]])
    return id_seq,freq_seq



def read_dicts():

    word_dict = Dict("data/word_dict")
    lemma_dict = Dict("data/lemma_dict")
    aux_dict = Dict("data/aux_dict")
    high_dict = Dict("data/high_dict")
    pos_dict = Dict("data/pos_dict")
    ner_dict = Dict("data/ner_dict")
    rel_dict = Dict("data/rel_dict")
    category_dict = Dict("data/category_dict")

    word_dict.load()
    lemma_dict.load()
    pos_dict.load()
    ner_dict.load()
    rel_dict.load()
    category_dict.load()
    high_dict.load()
    aux_dict.load()
    dicts = dict()

    dicts["rel_dict"] = rel_dict
    dicts["word_dict"] = word_dict
    dicts["pos_dict"] = pos_dict
    dicts["ner_dict"] = ner_dict
    dicts["lemma_dict"] = lemma_dict
    dicts["category_dict"] = category_dict
    dicts["aux_dict"] = aux_dict
    dicts["high_dict"] = high_dict
    return dicts

class Dict(object):
    def __init__(self, fileName,dictionary=None):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}

        # Special entries will not be pruned.
        self.special = []

        if dictionary :
            for label in dictionary:
                self.labelToIdx[label] = dictionary[label][0]
                self.idxToLabel[dictionary[label][0]] = label
                self.frequencies[dictionary[label][0]] = dictionary[label][1]
        self.fileName = fileName
                


    def size(self):
        return len(self.idxToLabel)
    
    def __len__(self):
        return len(self.idxToLabel)

    # Load entries from a file.
    def load(self, filename =None):
        if filename:
            self.fileName = filename
        else:
            filename = self.fileName 
        f = Pickle_Helper(filename) 
        data = f.load()
        self.idxToLabel=data["idxToLabel"]
        self.labelToIdx=data["labelToIdx"]
        self.frequencies=data["frequencies"]

    # Write entries to a file.
    def save(self, filename  =None):
        if filename:
            self.fileName = filename
        else:
            filename = self.fileName 
        f = Pickle_Helper(filename) 
        f.dump( self.idxToLabel,"idxToLabel")
        f.dump( self.labelToIdx,"labelToIdx")
        f.dump( self.frequencies,"frequencies")
        f.save()

    def lookup(self, key, default=None):
        try:
            return self.labelToIdx[key]
        except KeyError:
            if default: return default

            return self.labelToIdx[UNK_WORD]
    def __str__(self):
        out_str = []
        for k in self.frequencies:
            if k not in self.special:
                out_str.append(self.idxToLabel[k]+": "+str(self.frequencies[k]))
        return " \n".join(out_str)
    def __getitem__(self, label,default=None):
        try:
            return self.labelToIdx[label]
        except KeyError:
            if default: return default

            return self.labelToIdx[UNK_WORD]
    
    def getLabel(self, idx, default=UNK_WORD):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default
        
    def __iter__(self): return self.labelToIdx.__iter__()
    def __next__(self): return self.labelToIdx.__next__()
    # Mark this `label` and `idx` as special (i.e. will not be pruned).
    def addSpecial(self, label, idx=None):
        idx = self.add(label, idx)
        self.special += [idx]

    # Mark all labels in `labels` as specials (i.e. will not be pruned).
    def addSpecials(self, labels):
        for label in labels:
            self.addSpecial(label)

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label, idx=None):
        if idx is not None:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            if label in self.labelToIdx:
                idx = self.labelToIdx[label]
            else:
                idx = len(self.idxToLabel)
                self.idxToLabel[idx] = label
                self.labelToIdx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx
    
    def __setitem__(self, label, idx):
        self.add(label,idx)
        
        
    # Return a new dictionary with the `size` most frequent entries.
    def prune(self, size):
        if size >= self.size():
            return self

        # Only keep the `size` most frequent entries.
        freq = torch.Tensor(
                [self.frequencies[i] for i in range(len(self.frequencies))])
        _, idx = torch.sort(freq, 0, False)

        newDict = Dict(self.fileName)

        # Add special entries in all cases.
        for i in self.special:
            newDict.addSpecial(self.idxToLabel[i])

        for i in idx[:size]:
            newDict.add(self.idxToLabel[i])

        return newDict
    # Return a new dictionary with the `size` most frequent entries.
    def pruneByThreshold(self, threshold):
        # Only keep the `size` most frequent entries.
        high_freq = [ (self.frequencies[i],i) for i in range(len(self.frequencies)) if self.frequencies[i]>threshold]

        newDict = Dict(self.fileName)

        # Add special entries in all cases.
        for i in self.special:
            newDict.addSpecial(self.idxToLabel[i])

        for freq,i in high_freq:
            newDict.add(self.idxToLabel[i])
            newDict.frequencies[newDict.labelToIdx[self.idxToLabel[i]]] = freq

        return newDict
    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, labels, unkWord = UNK_WORD, bosWord=BOS_WORD, eosWord=EOS_WORD):
        vec = []

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        vec += [self.lookup(label, default=unk) for label in labels]

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        return torch.LongTensor(vec)

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convertToLabels(self, idx, stop=[]):
        labels = []

        for i in idx:
            if i in stop:
                break
            labels += [self.getLabel(i)]

        return labels