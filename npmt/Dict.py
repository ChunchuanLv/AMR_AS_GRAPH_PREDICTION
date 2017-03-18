import torch
from utility.constants import *
from utility.amr import *
from utility.pickle_helper import *

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

    def lookup(self, key, default=UNK):
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default
        
    def __getitem__(self, label,default =UNK):
        try:
            return self.labelToIdx[label]
        except KeyError:
            return default
    
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
        _, idx = torch.sort(freq, 0, True)

        newDict = Dict()

        # Add special entries in all cases.
        for i in self.special:
            newDict.addSpecial(self.idxToLabel[i])

        for i in idx[:size]:
            newDict.add(self.idxToLabel[i])

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
    def convertToLabels(self, idx, stop):
        labels = []

        for i in idx:
            labels += [self.getLabel(i)]
            if i == stop:
                break

        return labels