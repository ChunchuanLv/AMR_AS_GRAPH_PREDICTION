from utility.constants import *
from utility.pickle_helper import *
import torch
from torch.autograd import Variable
    

class DataIterator(object):

    def __init__(self, filePathes,total_size = 256,cuda = True,volatile = False ,dicts =None):
        self.totalSize = total_size
        self.cuda = cuda
        self.volatile = volatile
        self.concept_dict = dicts["concept_dict"]
        self.lemma_dict = dicts["lemma_dict"]
        concept_ls = dicts["concept_ls"]
        self.all = []
        
        self.src = []
        self.snt = []
        self.concept_ids = Variable(torch.LongTensor(concept_ls).view(1,len(concept_ls)))
        self.n_freq = len(concept_ls)
        self.concept_lut = torch.LongTensor(dicts["lemma_dict"].size()).fill_(PAD)
        for i, id_con in enumerate(concept_ls):  #chunck make tensor list
            self.concept_lut[id_con] = i
            
        if cuda:
            self.concept_lut = self.concept_lut.cuda()
            self.concept_ids = self.concept_ids.cuda()
            
        for filepath in filePathes:
            print(("reading "+filepath.split("/")[-1]+"......"))
            n = self.readFile(filepath)
            print(("done reading "+filepath.split("/")[-1]+", "+str(n)+" sentences processed"))
       #     break
            
        self.all = sorted(self.all, key=lambda x: x[0])

        self.tgt = []
        self.index = []
        for d in self.all:
            self.src.append(d[1])
            self.tgt.append(d[2])
            self.index.append(d[3])
        self.indexPair = []
        total_len = 0
        initial_index = 0
        for i,src in enumerate(self.src):
            length = src.size(0)
            if total_len + length > total_size:
                self.indexPair.append((initial_index,i))
                initial_index = i
                total_len = 0
            total_len += length
        self.indexPair.append((initial_index,len(self.src)))
        self.numBatches = len(self.indexPair)
            
    #return shape sentence_len, 4
    def read_sentence(self,data):
        self.all.append((len(data["snt_id"]),torch.LongTensor([data["snt_id"],data["lemma_id"],data["pos_id"],data["ner_id"]]).t().contiguous(),torch.LongTensor(data["amr_id"]).contiguous(),data["index"]))

            


    def readFile(self,filepath):
        data_file = Pickle_Helper(filepath.replace(".txt",".pickle")) 

        all_data = data_file.load()["data"]
        for data in all_data:
       #     print (data)
            self.read_sentence(data)
        return len(all_data)
    
    #out_each_step : tgt_len x sourceL x batch_size
    def _batchify(self, data,align_right=False, ):
        max_length = max(x.size(0) for x in data)
        out = data[0].new(len(data), max_length,data[0].size(1)).fill_(PAD)
        offsets = []
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            offsets.append(offset)
            out[i].narrow(0, offset, data_length).copy_(data[i])

        out = out.transpose(0,2).contiguous()
        if self.cuda:
            out = out.cuda()

        v = Variable(out,volatile = self.volatile)
        return v,offsets,max_length,out
    
    #high_index : tgt_len x batch_size        (0 ... n_high)
    #rule_index : tgt_len x batch_size x src_len  (0 1)
    def _batchifyId(self,data,offsets,src_len,tgt_len,tgtBatch):
        high_index = []
        rule_index = []
        if self.cuda:
            LongTensor = torch.cuda.LongTensor
        else:
            LongTensor = torch.LongTensor
        for t in range(tgt_len):
            high_index.append(self.concept_lut[tgtBatch[0][t]])
            rule_index.append([])
            for i in range(len(data)):
                rule_index[-1].append([0 for i in range(src_len)])
                if t < len(data[i]):
                    for index in data[i][t]:
                        if index == -2:
                            rule_index[-1][-1] = [1 for i in range(src_len)]
                            break
                        elif index != -1:
                            rule_index[-1][-1][index + offsets[i]-1] = 1
                else:
                    rule_index[-1][-1][src_len-1] = 1
        high_index_v = Variable(torch.stack(high_index),volatile = self.volatile )
        rule_index_v = Variable(torch.cuda.ByteTensor(rule_index),volatile = self.volatile) if self.cuda \
        else Variable(torch.ByteTensor(rule_index),volatile = self.volatile)

        return high_index_v,rule_index_v
        '''    print (self.n_freq ,tgtBatch[0])
        print ([self.concept_dict.getLabel(i) for i in tgtBatch[0,:,1]])
        print ([self.lemma_dict.getLabel(i) for i in tgtBatch[0,:,1]])
        
        print (lemma_index.sum(2).squeeze(2))
        print (high_index)'''
    def __getitem__(self, index):
        assert index <= self.numBatches, "%d > %d" % (index, self.numBatches)
        startId,endId = self.indexPair[index]
        srcBatch,offsets,max_length,Batch_t = self._batchify(
            self.src[startId:endId], align_right=True)

        if self.tgt:
            tgtBatch,offsetsY,max_lengthY,tgtBatch_t = self._batchify(
                self.tgt[startId:endId],align_right=False)
            idBatch = self._batchifyId(self.index[startId:endId],offsets,max_length,max_lengthY,tgtBatch_t)
        else:
            tgtBatch = None
            idBatch = None
    #    print ("high slot filled?",(idBatch[0].ne(PAD).float()).min())
    #    print ("lemma slot filled?",(idBatch[1].sum(2).ne(0).float()).min())
    #    print ("all slot filled?",(idBatch[0].ne(PAD).float()+idBatch[1].sum(2).ne(0).float()).min())
    #    print ("srcBatch",srcBatch.size())
  #      print ("in data_iterator tgtBatch",tgtBatch.size())
    #    print ("idBatch",idBatch[0].size(),idBatch[1].size())

   #     del offsets,offsetsY,Batch_t,tgtBatch_t
        tgtBatch = tgtBatch[0:2]
        return srcBatch, tgtBatch, idBatch

    def __len__(self):
        return self.numBatches
    
if __name__ == "__main__":
    training_data = DataIterator(trainingFilesPath,batchSize = 2,cuda = True,training = True)
    srcBatch, tgtBatch, idBatch = training_data[0]
    i = 6
    print (srcBatch, tgtBatch, idBatch)