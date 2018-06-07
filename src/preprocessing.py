#!/usr/bin/env python3.6
# coding=utf-8
'''

Combine multiple AMR data files in the same directory into a single one
Need to specify folder containing all subfolders of training, dev and test

Then extract features for futher process based on stanford core nlp tools

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-06-01
'''

from parser.AMRProcessors import *

import argparse


def combine_files(files):
    out = "/".join(files[0].split("/")[:-1])
    out = out + "/combined.txt_"
    with open(out, 'w+') as outfile:
        for fname in files:
            with open(fname) as infile:
                line = infile.readline()
                line = infile.readline()
                while line != '' :
                    line = infile.readline()
                    outfile.write(line)
                outfile.write("\n")

def write_features(filepath,feature_extractor:AMRInputPreprocessor):
    out = filepath + "pre_processed"
    print ("processing "+filepath)
    n = 0
    with open(out,'w') as out_f:
        with open(filepath,'r') as f:
            line = f.readline()
            while line != '' :
                if line.startswith("# ::snt") or line.startswith("# ::tok"):
                    text = line[7:]
                    data = feature_extractor.preprocess(text)
                    out_f.write(line.replace("# ::tok","# ::snt"))
                    for key in ["tok","lem","pos","ner"]:
                        out_f.write("# ::"+key+"\t"+"\t".join(data[key])+"\n")
                    n = n+1
                    if n % 500 ==0:
                        print (str(n)+" sentences processed")
                elif not line.startswith("# AMR release; "):
                    out_f.write(line)
                line = f.readline()
    print ("done processing "+filepath)
    print (out +" is generated")

def combine_arg():
    parser = argparse.ArgumentParser(description='preprocessing.py')

    ## Data options
    parser.add_argument('-suffix', default="txt", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('-folder', default=allFolderPath, type=str ,
                        help="""the folder""")
    return parser


parser = combine_arg()


opt = parser.parse_args()
feature_extractor = AMRInputPreprocessor()

trainFolderPath = opt.folder+"/training/"
trainingFilesPath = folder_to_files_path(trainFolderPath,opt.suffix)
combine_files(trainingFilesPath)
write_features(trainFolderPath+"/combined.txt_",feature_extractor)

devFolderPath = opt.folder+"/dev/"
devFilesPath = folder_to_files_path(devFolderPath,opt.suffix)
combine_files(devFilesPath)
write_features(devFolderPath+"/combined.txt_",feature_extractor)

testFolderPath = opt.folder+"/test/"
testFilesPath = folder_to_files_path(testFolderPath,opt.suffix)
combine_files(testFilesPath)
write_features(testFolderPath+"/combined.txt_",feature_extractor)

