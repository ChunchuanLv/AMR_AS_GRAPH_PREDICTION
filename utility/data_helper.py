#!/usr/bin/env python3.6
# coding=utf-8
'''

Some helper functions for storing and reading data

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-29
'''
import json,os,re
import pickle


class Pickle_Helper:

    def __init__(self, filePath):
        self.path = filePath
        self.objects = dict()

    def dump(self,obj,name):
        self.objects[name] = obj

    def save(self):
        f = open(self.path , "wb")
        pickle.dump(self.objects ,f,protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load(self):
        f = open(self.path , "rb")
        self.objects = pickle.load(f)
        f.close()
        return self.objects
    
    def get_path(self):
        return self.path
    
class Json_Helper:

    def __init__(self, filePath):
        self.path = filePath
        self.objects = dict()

    def dump(self,obj,name):
        self.objects[name] = obj

    def save(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        for name in self.objects:
            with open(self.path+"/"+name+".json", 'w+') as fp:
                json.dump(self.objects[name], fp)

    def load(self):
        files_path = folder_to_files_path(self.path,ends =".json")
        for f in files_path:
            name = f.split("/")
            with open(f) as data_file:
                data = json.load(data_file)
                self.objects[name] = data
        return self.objects

    def get_path(self):
        return self.path

def folder_to_files_path(folder,ends =".txt"):
    files = os.listdir(folder )
    files_path = []
    for f in files:
        if f.endswith(ends):
            files_path.append(folder+f)
          #  break
    return   files_path
def load_line(line,data):
    if "\t" in line:
        tokens = line[4:].split("\t")
    else:
        tokens = line[4:].split()
    if tokens[0] == "root": return

    if tokens[0] == "node":
        data["node"][tokens[1]] = tokens[2]
        if tokens.__len__() > 3:
            data["align"][tokens[1]] = int(tokens[3].split("-")[0])
        return
    if tokens[0] == "edge":
        data["edge"][tokens[4],tokens[5]] = tokens[2]
        return
    data[tokens[0]] = tokens[1:]
def asserting_equal_length(data):
    assert len(data["tok"]) ==len(data["lem"]) , (  len(data["tok"]) ,len(data["lem"]),"\n",list(zip(data["tok"],data["lem"])) ,data["tok"],data["lem"])
    assert len(data["tok"]) ==len(data["ner"]) , (  len(data["tok"]) ,len(data["ner"]),"\n",list(zip(data["tok"],data["ner"])) ,data["tok"],data["ner"])
    assert len(data["tok"]) ==len(data["pos"]) , (  len(data["tok"]) ,len(data["pos"]),"\n",list(zip(data["tok"],data["pos"])) ,data["tok"],data["pos"])

def load_text_jamr(filepath):
    all_data = []
    with open(filepath,'r') as f:
        line = f.readline()
        while line != '' :
            while line != '' and not line.startswith("# ::") :
                line = f.readline()

            if line == "": return all_data

            data = {}
            data.setdefault("align",{})
            data.setdefault("node",{})
            data.setdefault("edge",{})
            while line.startswith("# ::"):
                load_line(line.replace("\n","").strip(),data)
                line = f.readline()
            amr_t = ""
            while line.strip() != '' and not line.startswith("# AMR release"):
                amr_t = amr_t+line
                line = f.readline()
            data["amr_t"] = amr_t
            asserting_equal_length(data)
            all_data.append(data)
            line = f.readline()
    return all_data


def load_text_input(filepath):
    all_data = []
    with open(filepath,'r') as f:
        line = f.readline()
        while line != '' :
            while line != '' and not line.startswith("# ::"):
                line = f.readline()

            if line == "": return all_data

            data = {}
            while line.startswith("# ::"):
                load_line(line.replace("\n","").strip(),data)
                line = f.readline()
            all_data.append(data)
            line = f.readline()
    return all_data