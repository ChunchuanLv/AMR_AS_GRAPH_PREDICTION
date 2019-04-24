#!/usr/bin/env python3.6
# coding=utf-8
'''

AMRParser for producing amr graph from raw text
AMRDecoder for decoding deep learning model output into actual AMR concepts and graph
AMRInputPreprocessor for extract features based on stanford corenlp
@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''
from torch.autograd import Variable
from utility.StringCopyRules import  *
from utility.ReCategorization import  *
from parser.Dict import seq_to_id
from utility.constants import core_nlp_url
import networkx as nx

from src import *
from parser.modules.helper_module import myunpack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import PackedSequence
from parser.DataIterator import DataIterator,rel_to_batch
from pycorenlp import StanfordCoreNLP
class AMRInputPreprocessor(object):
    def __init__(self, url = core_nlp_url):
        self.nlp = StanfordCoreNLP(url)
        self.joints_map = self.readJoints()
        self.number_texts = {"hundred", "thousand", "million", "billion", "trillion", "hundreds", "thousands",
			"millions", "billions", "trillions"}
        self.slashedNumber =  re.compile(r'-*\d+-\d+')

    def readJoints(self):
        joints_map = {}
        with open("data/joints.txt",'r') as f:
            line = f.readline()
            while line.strip() != '':
                line = f.readline()
                compounds = line.split()
                past = ""
                for w in compounds:
                    joints_map.setdefault(past[:-1],[]).append(w)
                    past = past + w + "-"
        return joints_map

    def combine_number(self,data):
    #combine phrase e.g. :  make up
        def combinable_number(n1,n2):
            return n2 in self.number_texts and n1 != "-"
        def combinable(i,m):
            return len(lemma) > 0 and m == "CD"\
                    and pos[-1] =="CD" and combinable_number(lemma[-1], data["lem"][i])
        lemma = []
        ner = []
        tok = []
        pos = []

        for i, m in enumerate(data["pos"]):
            if combinable(i,m) :
                lemma[-1] = lemma[-1] +"," + data["lem"][i]
                tok[-1] = tok[-1] + "," + data["tok"][i]
                pos[-1] = "CD"
        #        ner[-1] = ner[-1]
            else:
                lemma.append(data["lem"][i])
                tok.append(data["tok"][i])
                pos.append(data["pos"][i])
                ner.append(data["ner"][i])

        data["lem"] = lemma
        data["ner"] = ner
        data["pos"] = pos
        data["tok"] = tok
        return data

    def tag_url_and_split_number(self,data):
        lemma = []
        ner = []
        tok = []
        pos = []

        for i, le in enumerate(data["lem"]):
            if "http" in le or "www." in le:
                ner.append("URL")
                lemma.append(data["lem"][i])
                tok.append(data["tok"][i])
                pos.append(data["pos"][i])

            elif re.match(self.slashedNumber, le) and data["ner"][i] == "DATE":
                les = le.replace("-"," - ").split()
                toks = data["tok"][i].replace("-"," - ").split()
                assert len(les) == len(toks),data
                for  l in les:
                    if l != "-":
                        pos.append(data["pos"][i])
                        ner.append(data["ner"][i])
                    else:
                        pos.append(":" )
                        ner.append("0")
                lemma = lemma + les
                tok = tok + toks
            else:
                ner.append(data["ner"][i])
                lemma.append(data["lem"][i])
                tok.append(data["tok"][i])
                pos.append(data["pos"][i])

        data["lem"] = lemma
        data["ner"] = ner
        data["pos"] = pos
        data["tok"] = tok
        return data

    def combine_phrase(self,data):
    #combine phrase e.g. :  make up
        lemma = []
        ner = []
        tok = []
        pos = []
        skip = False
        for i ,le in enumerate(data["lem"]):
            if skip:
                skip = False
            elif len(lemma) > 0 and le in self.joints_map.get( lemma[-1] ,[]) :
                    lemma[-1] = lemma[-1] +"-"+le
                    tok[-1] = tok[-1]+ "-" + data["tok"][i]
                    pos[-1] = "COMP"
                    ner[-1] = "0"
            elif len(lemma) > 0 and le == "-" and i < len(data["lem"])-1 \
                and data["lem"][i+1] in self.joints_map.get( lemma[-1] ,[]):
                lemma[-1] = lemma[-1] +"-"+data["lem"][i+1]
                tok[-1] = tok[-1]+ "-" + data["tok"][i+1]
                pos[-1] = "COMP"
                ner[-1] = "0"
                skip = True
            else:
                lemma.append(le)
                tok.append(data["tok"][i])
                pos.append(data["pos"][i])
                ner.append(data["ner"][i])

        data["lem"] = lemma
        data["ner"] = ner
        data["pos"] = pos
        data["tok"] = tok
        return data
    def featureExtract(self,src_text,whiteSpace=False):
        data = {}
        output = self.nlp.annotate(src_text.strip(), properties={
        'annotators': "tokenize,ssplit,pos,lemma,ner",
        "tokenize.options":"splitHyphenated=true,normalizeParentheses=false",
		"tokenize.whitespace": whiteSpace,
        'ssplit.isOneSentence': True,
        'outputFormat': 'json'
    })
        snt = output['sentences'][0]["tokens"]
        data["ner"] = []
        data["tok"] = []
        data["lem"] = []
        data["pos"] = []
        for snt_tok in snt:
            data["ner"].append(snt_tok['ner'])
            data["tok"].append(snt_tok['word'])
            data["lem"].append(snt_tok['lemma'])
            data["pos"].append(snt_tok['pos'])
     #   if whiteSpace is False:
     #       return self.featureExtract(" ".join(data["tok"]),True)
        asserting_equal_length(data)
        return data

    def preprocess(self,src_text):
        data = self.featureExtract(src_text)
        data = self.combine_phrase(data) #phrase from fixed joints.txt file
        data = self.combine_number(data)
        data = self.tag_url_and_split_number(data)
        asserting_equal_length(data)
        return data

class AMRParser(object):
    def __init__(self, opt,dicts,parse_from_processed=False):
        self.decoder = AMRDecoder(opt,dicts)
        self.model = load_old_model(dicts,opt,True)[0]
        self.opt = opt
        self.parse_from_processed = parse_from_processed
        if not parse_from_processed:
            self.feature_extractor = AMRInputPreprocessor()
        self.dicts = dicts
        self.decoder.eval()
        self.model.eval()

    def feature_to_torch(self,all_data):
        for i, data in enumerate(all_data):
            data["snt_id"] = seq_to_id(self.dicts ["word_dict"],data["tok"])[0]
            data["lemma_id"] = seq_to_id(self.dicts ["lemma_dict"],data["lem"])[0]
            data["pos_id"] =  seq_to_id(self.dicts ["pos_dict"],data["pos"])[0]
            data["ner_id"] = seq_to_id(self.dicts ["ner_dict"],data["ner"])[0]
        data_iterator = DataIterator([],self.opt,self.dicts["rel_dict"],volatile = True,all_data=all_data)
        order,srcBatch,src_sourceBatch = data_iterator[0]
        return order,srcBatch,src_sourceBatch,data_iterator

    def parse_batch(self,src_text_batch_or_data_batch):
        if not self.parse_from_processed:
            all_data =[ self.feature_extractor.preprocess(src_text) for src_text in src_text_batch_or_data_batch ]
        else:
            all_data = src_text_batch_or_data_batch
        order,srcBatch,sourceBatch,data_iterator = self.feature_to_torch(all_data)
        probBatch = self.model(srcBatch )

        amr_pred_seq,concept_batches,aligns_raw,dependent_mark_batch = self.decoder.probAndSourceToAmr(sourceBatch,srcBatch,probBatch,getsense = True )

        amr_pred_seq = [ [(uni.cat,uni.le,uni.aux,uni.sense,uni)  for uni in seq ] for  seq in amr_pred_seq ]


        rel_batch,aligns = rel_to_batch(concept_batches,aligns_raw,data_iterator,self.dicts)
        rel_prob,roots = self.model((rel_batch,srcBatch,aligns),rel=True)
        graphs,rel_triples  =  self.decoder.relProbAndConToGraph(concept_batches,rel_prob,roots,(dependent_mark_batch,aligns_raw),True,True)
        batch_out = [0]*len(graphs)

        for i,data in enumerate(zip( sourceBatch,amr_pred_seq,concept_batches,rel_triples,graphs)):
            source,amr_pred,concept, rel_triple,graph= data
            predicated_graph = graph_to_amr(graph)

            out = []
            out.append( "# ::tok "+" ".join(source[0])+"\n")
            out.append(  "# ::lemma "+" ".join(source[1])+"\n")
            out.append(  "# ::pos "+" ".join(source[2])+"\n")
            out.append(  "# ::ner "+" ".join(source[3])+"\n")
            out.append(  self.decoder.nodes_jamr(graph))
            out.append(  self.decoder.edges_jamr(graph))
            out.append( predicated_graph)
            batch_out[order[i]] = "".join(out)+"\n"
        return batch_out

    def parse_one(self,src_text):
        return self.parse_batch([src_text])


def mark_edge(n1,n2,edge,edges):
    edges.add((n1,n2,edge))
    inverse = edge[:-3] if edge.endswith("-of") else edge+"-of"
    edges.add((n2,n1,inverse))

# ::node	0.1.1.1.1	kilometer	26-27
def sub_to_amr(G,visited,n,visited_edges,d=1):
    uni = G.node[n]["value"]
    if uni.is_constant():
        if uni.cat == Rule_String:
            return "\""+ uni.le+"\"",visited,visited_edges
        return uni.le,visited,visited_edges
    s = "("+str(n)+" /"+" "+ uni.le+uni.sense

    to_be_visted_edge = set()  #to_be_visted_edges around this node
    to_be_visted = set() #to be visited nodes around this node
    for nb,edge_data in G[n].items():
        if not nb in visited :
            to_be_visted.add(nb)
        if (n,nb,edge_data["role"]) not in visited_edges:
            mark_edge(n,nb,edge_data["role"],to_be_visted_edge)
    for nb,edge_data in G[n].items():
        if (n,nb,edge_data["role"]) in to_be_visted_edge:
            forced = "force_connect" in edge_data
            append =  "" # "!" if  forced else ""
            mark_edge(n,nb,edge_data["role"],visited_edges)
            s += "\n"+"    "*d+edge_data["role"]+append+" "
            if nb in visited and nb not in to_be_visted:
                uni = G.node[nb]["value"]
                if uni.is_constant():
                    if uni.cat == Rule_String:
                        s +=  "\""+ uni.le+"\""
                    else:
                        s +=  uni.le
                else:
                    s += str(nb)
            else:
                ss,visited,visited_edges = sub_to_amr(G,visited.union(to_be_visted),nb, visited_edges.union(to_be_visted_edge),d+1)
                s += ss
    s += ")"
    return s,visited,visited_edges

def my_contracted_nodes(G,n1,n2,self_loops=False):
    d = G.node[n1]
 #   print ("before",G.node[n1])
    G = nx.contracted_nodes(G, n1,n2,self_loops=self_loops)
 #   print ("after",G.node[n1])
    for k in d:
        G.node[n1][k] = d[k]
 #   print ("finally",G.node[n1])
    return G

#collapsing person nodes when its' likely its' the same person being activated by NER and other role.
#e.g. President Donald Trump might evoke person node twice.
def contract_graph(G):
    NAMES = {}
    for n in G.node:
        if not G.node[n]["value"].cat == Rule_Frame:
            continue
        person = [nb for nb,edge in G[n].items()
                  if G.node[nb]["value"].le =="person" and  edge["role"] == ":ARG0" ]
        if len(person) < 2: continue
        p1 = person[0]
        for pi in  person[1:]:
            G = my_contracted_nodes(G,p1,pi)
        return contract_graph(G)


    for n in G.node:
        if not G.node[n]["value"].le == "name":
            continue
        top = None
        topn = None
        names = []
        namesn = []
        for nb,edge in G[n].items():
            if edge["role"] == ":name-of":
                top = G.node[nb]["value"]
                topn = nb
            if edge["role"].startswith(":op"):
                names.append((edge["role"][2:],G.node[nb]["value"]))
                namesn.append((edge["role"][2:],nb))
        ner =  (top,tuple(sorted(names,key=lambda x: x[0])))
        namesn = sorted(namesn,key=lambda x: x[0])
        if ner in NAMES:
            G = my_contracted_nodes(G, NAMES[ner][0],n)
            if top is None: continue
            G = my_contracted_nodes(G, NAMES[ner][1],topn)
            for s1,s2 in zip(namesn,NAMES[ner][2]):
                G = my_contracted_nodes(G,s2[1],s1[1])
            return contract_graph(G)
        else:
            NAMES[ner] = (n,topn,namesn)
    for n in G.node:
        for n1,edge1 in G[n].items():
            for n2,edge2 in G[n].items():
                if str(n1) < str(n2) and G.node[n1]["value"] ==  G.node[n2]["value"] \
                    and edge1["role"] ==  edge2["role"] and G.node[n1]["value"].is_constant():
                    return contract_graph(my_contracted_nodes(G, n1,n2))
    return G
#Turning data structure to AMR text for evaluation
def graph_to_amr(G, sentence = None):
    top = BOS_WORD
    root = list(G[top].keys())[0]
    uni = G.node[root]["value"]
    if uni.is_constant():
        G.node[root]["value"].cat = Rule_Concept
        G.node[root]["value"].le = G.node[root]["value"].le.strip(":").strip("/")
        if  G.node[root]["value"].le == "":
            G.node[root]["value"].le = "amr-unintelligible"
    s,visited,visited_edges =  sub_to_amr(G,{top,root},root,{(top,root,":top"),(root,top,":top-of")})
    for n in G.node:
        if not  n in visited:
            print ("node missing",n)
            print ("nearbour",G.node[n])
            for n in G.node:
                print ("nodes",n,G.node[n],G.node[n]["value"])
            print ("visited nodes",visited)
            print ("edges",G.edges)
            print ("visited edges",visited_edges)
            print (s,sentence)
            break
      #      assert False
    for e in G.edges:
        if not  (e[0],e[1],G.edges[e]["role"]) in visited_edges:
            print ("edge missing",e[0],e[1],G.edges[e]["role"])
            print ("all nodes",G.nodes)
            print ("visited nodes",visited)
            print ("edges",G.edges)
            print ("visited edges",visited_edges)
            print (s,sentence)
            break
    return s
from utility.amr import is_core

#decoder requires copying dictionary, and recategorization system
#full_tricks evoke some deterministic post processing, which might be expansive to compute
class AMRDecoder(object):
    def __init__(self, opt,dicts):
        self.opt = opt
        self.dicts = dicts
        self.n_high = len(dicts["high_dict"])
        self.rl = rules()
        with_jamr = "_with_jamr" if opt.jamr else "_without_jamr"
        self.rl.load("data/rule_f"+with_jamr)
        self.fragment_to_node_converter = ReCategorizor(from_file=True,path="data/graph_to_node_dict_extended"+with_jamr,training=False,ner_cat_dict=dicts["aux_dict"])
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def getMaskAndLengths(self,batch):
        #srcBatch: len, batch, n_feature
        lengths = []

        if isinstance(batch, tuple):
            batch = batch[0]
        leBatch = batch[:,:,0].transpose(0,1).data.tolist()
        for i in range(len(leBatch)):
            l = leBatch[i].index(PAD) if PAD in  leBatch[i] else len(leBatch[i])
            lengths.append(l)
        leBatch = batch[:,:,TXT_LEMMA]
        if isinstance(leBatch,Variable):
            leBatch = leBatch.data
        mask = leBatch != PAD

        return mask,leBatch,lengths


    def probToId(self, srcBatch,probBatch,sourceBatch):
        #batch of id
        out = []
        assert isinstance(srcBatch, PackedSequence)

        srcBatch = unpack(srcBatch)[0]
        mask ,leBatch,lengths= self.getMaskAndLengths(srcBatch)
        mask = mask.unsqueeze(2)
        for i,prob in enumerate(probBatch):
            assert isinstance(prob, tuple)
            prob = unpack(prob)[0]
            if i== AMR_LE:
                prob = prob.data  #    src_len x batch xn_out
                n_out = prob.size(2)
                best,max_indices, = prob.max(dim=2,keepdim=True)
                assert max_indices.size(-1) == 1,("start",i, best,max_indices)
                h = (max_indices==n_out).long()*srcBatch[:,:,TXT_LEMMA].data.unsqueeze(2)
                assert h.size(-1) == 1,("middle",i, best.size(),h.size(),max_indices.size(),srcBatch[:,:,TXT_LEMMA].data.size())
                max_indices = max_indices*(max_indices<n_out).long()+h
                assert max_indices.size(-1) == 1,("middle",i, best.size(),max_indices.size())
                max_indices = max_indices*mask.long()
                assert max_indices.size(-1) == 1,("end",i, best.size(),max_indices.size(),mask.size())
                out.append(max_indices)
            else:
                prob = prob.data  #    src_len x batch xn_out
                best,max_indices, = prob.max(dim=2,keepdim=True)
                out.append(max_indices)
        l = out[0].size(2)
        for d in out:
            assert d.size(2) ==l, ([dd.size() for  dd in out],"\n",srcBatch,probBatch,"\n",[" ".join(src[0])+"\n" for src in sourceBatch])
        return torch.cat(out,2),lengths

    def probAndSourceToAmr(self, sourceBatch,srcBatch,probBatch,getsense=False):
        #batch of id
        #max_indices len,batch, n_feature
        #srcBatch  batch source
        #out batch AMRuniversal
        def id_to_high(max_le_id):
            out = []
            for id in max_le_id:
                if id < self.n_high:
                    out.append( self.dicts["high_dict"].getLabel(id))
                else:
                    out.append(None)
            return out
        max_indices, lengths = self.probToId(srcBatch,probBatch,sourceBatch)
        out = []
        srl_concepts = []
        dependent_mark_batch = []
        aux_triples_batch = []
        aligns = []
        for i,source in enumerate(sourceBatch):
            snt,lemma = source[0:2]
            ners = source[3]
            max_id = max_indices[:,i,:]
            cats = self.dicts["category_dict"].convertToLabels(max_id[:,AMR_CAT])
            aux = self.dicts["aux_dict"].convertToLabels(max_id[:,AMR_NER])
            high = id_to_high(max_id[:,AMR_LE] )
      #      high = max_indices==n_out if
            if AMR_SENSE< max_id.size(1) and False:
                sense = self.dicts["aux_dict"].convertToLabels(max_id[:,AMR_SENSE])
            else:
                sense = None
            amr_seq = self.rl.toAmrSeq(cats,snt,lemma,high,aux,sense,ners )
            srl_concept,align,dependent_mark = self.fragment_to_node_converter.unpack_recategorized(amr_seq,self.rl ,getsense,eval= not self.training)

            if len(srl_concept) == 0 :
                srl_concept = [AMRUniversal("amr-unintelligible",Rule_Concept,None)]
                align = [0]
                dependent_mark = [0]
                aux_triples = []
       #         print (amr_seq,source)
       #         print ()


            srl_concepts.append(srl_concept)
            aligns.append(align)
            out.append(amr_seq)
            dependent_mark_batch.append(dependent_mark)

        return out,srl_concepts,aligns,dependent_mark_batch


    def graph_to_quadruples(self,graph):
        def add_to_quadruples(h_v,d_v,r,r_inver):
            if is_core(r):
                quadruples.append([graph.node[h_v]['value'],graph.node[d_v]['value'],r,h_v,d_v])
            else:
                quadruples.append([graph.node[d_v]['value'],graph.node[h_v]['value'],r_inver,d_v,h_v])
        quadruples = []

        for n,d in graph.nodes(True):
            if d["value"].le == BOS_WORD:
                continue
            for nearb in graph[n]:
                if graph[n][nearb]["role"] in self.dicts["rel_dict"] :
                    r = graph[n][nearb]["role"]
                    quadruples.append([graph.node[n]['value'],graph.node[nearb]['value'],r,n,nearb])
                elif  graph[n][nearb]["role"] == ":top-of":
                    quadruples.append([graph.node[nearb]['value'],graph.node[n]['value'],":top",nearb,n])

        return quadruples

    def graph_to_concepts(self,graph):
        concepts = []

        for n,d in graph.nodes(True):
            if d["value"].le == BOS_WORD:
                continue
            add = True
            for nearb in graph[n]:
                if ":wiki-of" in graph[n][nearb]["role"] :
                    add = False
                    break
            if add:
                concepts.append(d["value"])
        return concepts

    def graph_to_concepts_batches(self,graphs):
        return [self.graph_to_concepts(graph) for graph in graphs]

    def relProbAndConToGraph(self,srl_batch,srl_prob,roots,appended,get_sense=False,set_wiki=False):
        #batch of id
        #max_indices len,batch, n_feature
        #srcBatch  batch source
        #out batch AMRuniversal
        rel_dict = self.dicts["rel_dict"]
        def get_uni_var(concepts,id):
            assert id < len(concepts),(id,concepts)
            uni = concepts[id]
            if  uni.le in [ "i" ,"it","you","they","he","she"] and uni.cat == Rule_Concept:
                return uni,Var(uni.le )
            le = uni.le
            if uni.cat != Rule_String:
                uni.le = uni.le.strip("/").strip(":")
                if ":" in uni.le or "/" in uni.le:
                    uni.cat = Rule_String
            if uni.le == "":
                return uni,Var(le+ str(id))
            return uni,Var(uni.le[0]+ str(id))
        def create_connected_graph(role_scores,concepts,root_id,dependent,aligns):
            #role_scores: amr x amr x rel
            graph = nx.DiGraph()
            n = len(concepts)
            role_scores = role_scores.view(n,n,-1).data
            max_non_score, max_non_score_id= role_scores[:,:,1:].max(-1)
            max_non_score_id = max_non_score_id +1
            non_score = role_scores[:,:,0]
            active_cost =   non_score - max_non_score  #so lowest cost edge gets to active first
            candidates = []
            h_vs = []
            for h_id in range(n):
                h,h_v = get_uni_var(concepts,h_id)
                h_vs.append(h_v)
                graph.add_node(h_v, value=h, align=aligns[h_id],gold=True,dep = dependent[h_id])

            constant_links = {}
            normal_edged_links = {}
            for h_id in range(n):
                for d_id in range(n):
                    if h_id != d_id:
                        r = rel_dict.getLabel(max_non_score_id[h_id,d_id])
                        r_inver = r + "-of" if not r.endswith( "-of") else r[:-3]
                        h,h_v = get_uni_var(concepts,h_id)
                        d,d_v = get_uni_var(concepts,d_id)
                        if  (concepts[h_id].is_constant() or concepts[d_id].is_constant() ):
                            if concepts[h_id].is_constant() and concepts[d_id].is_constant() :
                                continue
                            elif concepts[h_id].is_constant():
                                constant_links.setdefault(h_v,[]).append((active_cost[h_id,d_id],d_v,r,r_inver))
                            else:
                                constant_links.setdefault(d_v,[]).append((active_cost[h_id,d_id],h_v,r_inver,r))
                        elif active_cost[h_id,d_id] < 0:
                            if r in [":name-of" ,":ARG0"] and concepts[d_id].le in ["person"]:
                                graph.add_edge(h_v, d_v, role=r)
                                graph.add_edge(d_v, h_v, role=r_inver)
                            else:
                         #       if concepts[h_id].le == "name" and r != ":name-of":
                          #          r = ":name-of"
                                normal_edged_links.setdefault((h_v,r),[]).append((active_cost[h_id,d_id],d_v,r_inver))
                        else:
                            candidates.append((active_cost[h_id,d_id],(h_v,d_v,r,r_inver)))

            max_edge_per_node = 1 if not self.training else 100
            for h_v,r in normal_edged_links:
                sorted_list = sorted(normal_edged_links[h_v,r],key = lambda j:j[0])
                for _,d_v,r_inver in sorted_list[:max_edge_per_node]:
               #     if graph.has_edge(h_v, d_v):
               #         continue
                    graph.add_edge(h_v, d_v, role=r)
                    graph.add_edge(d_v, h_v, role=r_inver)
                for cost,d_v,r_inver in sorted_list[max_edge_per_node:]:  #remaining
                    candidates.append((cost,(h_v,d_v,r,r_inver)))


            for h_v in constant_links:
                _,d_v,r,r_inver = sorted(constant_links[h_v],key = lambda j:j[0])[0]
                graph.add_edge(h_v, d_v, role=r)
                graph.add_edge(d_v, h_v, role=r_inver)

            candidates = sorted(candidates,key = lambda j:j[0])

            for _,(h_v,d_v,r,r_inver ) in candidates:
                if  nx.is_strongly_connected(graph):
                    break
                if not nx.has_path(graph,h_v,d_v):
                    graph.add_edge(h_v, d_v, role=r,force_connect=True)
                    graph.add_edge(d_v, h_v, role=r_inver,force_connect=True)

            _,root_v  = get_uni_var(concepts,root_id)
            h_v = BOS_WORD
            root_symbol = AMRUniversal(BOS_WORD,BOS_WORD,NULL_WORD)
            graph.add_node(h_v, value=root_symbol, align=-1,gold=True,dep=1)
            graph.add_edge(h_v, root_v, role=":top")
            graph.add_edge(root_v, h_v, role=":top-of")

            if get_sense:
                for n,d in graph.nodes(True):
                    if "value" not in d:
                        print (n,d, graph[n],constant_links,graph.nodes,graph.edges)
                    le,cat,sense = d["value"].le,d["value"].cat,d["value"].sense
                    if cat == Rule_Frame and sense == "":
                        sense = self.fragment_to_node_converter.get_senses(le)
                        d["value"] = AMRUniversal(le,cat,sense)

          #  if not self.training:
            #    assert nx.is_strongly_connected(graph),("before contraction",self.graph_to_quadruples(graph),graph_to_amr(graph))
           #     graph = contract_graph(graph)
            #    assert nx.is_strongly_connected(graph),("before contraction",self.graph_to_quadruples(graph),graph_to_amr(graph))

            if set_wiki:
                list = [[n,d]for n,d in graph.nodes(True)]
                for n,d in list:
                    for nearb in graph[n]:
                        r = graph[n][nearb]["role"]
                    if d["value"].le == "name":
                        names = []
                        head = None
                        for nearb in graph[n]:
                            r = graph[n][nearb]["role"]
                            if ":op" in r and "-of" not in r and  int(graph[n][nearb]["role"][3:]) not in names:
                                names.append([graph.node[nearb]["value"], int(graph[n][nearb]["role"][3:])])
                            if r == ":name-of":
                                wikied = False
                                for nearbb in graph[nearb]:
                                    r = graph[nearb][nearbb]["role"]
                                    if r == ":wiki":
                                        wikied = True
                                        break
                                if not wikied:
                                    head = nearb
                        if head:
                            names = tuple([t[0] for t in sorted(names,key = lambda t: t[1])])
                            wiki = self.fragment_to_node_converter.get_wiki(names)
                        #    print (wiki)
                            wiki_v = Var(wiki.le+n._name )
                            graph.add_node(wiki_v, value=wiki, align=d["align"],gold=True,dep=2) #second order  dependency
                            graph.add_edge(head, wiki_v, role=":wiki")
                            graph.add_edge(wiki_v, head, role=":wiki-of")

            assert nx.is_strongly_connected(graph),("after contraction",self.graph_to_quadruples(graph),graph_to_amr(graph))
            return graph,self.graph_to_quadruples(graph)

        graphs = []
        quadruple_batch = []
        score_batch = myunpack(*srl_prob) #list of (h x d)


        depedent_mark_batch = appended[0]
        aligns_batch = appended[1]
        for i,(role_scores,concepts,roots_score,dependent_mark,aligns) in enumerate(zip(score_batch,srl_batch,roots,depedent_mark_batch,aligns_batch)):
            root_s,root_id = roots_score.max(0)
            assert roots_score.size(0) == len(concepts),(concepts,roots_score)
            root_id = root_id.data.tolist()[0]
            assert root_id < len(concepts),(concepts,roots_score)

            g,quadruples = create_connected_graph(role_scores,concepts,root_id,dependent_mark,aligns)
            graphs.append(g)
            quadruple_batch.append(quadruples)

        return graphs,quadruple_batch

    def nodes_jamr(self,graph):
        s = []
        for n,d in graph.nodes(True):
            if d["value"].le == BOS_WORD:
                continue
            a = d["align"]
            assert isinstance(a,int),(n,a,graph.nodes(True))
            if a > -1:
                s.append("# ::node\t"+n._name+"\t"+d["value"].gold_str()+"\t"+ str(a)+"-"+str(a+1)+"\n") #+"\t"+str(d["dep"])
        return "".join(s)

# ::edge	border-01	ARG2	country	0.1.0	0.1.0.2
    def edges_jamr(self,graph,dep_only=False):
        s = []
        def cannonical(r):
            return  "-of" in r and not is_core(r) or  "-of"  not in r and  is_core(r)
        for n,d in graph.nodes(True):
            if d["value"].le == BOS_WORD:
                continue
            if dep_only:
                for nearb in graph[n]:
                    if graph.node[nearb]["dep"] > d["dep"] and  graph.node[nearb]["value"].le != BOS_WORD:
                        head = graph.node[n]["value"].gold_str()
                        dep = graph.node[nearb]["value"].gold_str()
                        r = graph[n][nearb]["role"]
                        assert isinstance(n,Var),(n,graph.nodes(True))
                        assert isinstance(nearb,Var),(nearb,graph.nodes(True))
                        s.append("# ::edge\t"+head+"\t"+ r+"\t"+dep+"\t"+n._name+"\t"+nearb._name+"\t"+"\n")
            else:
                for nearb in graph[n]:
                    r = graph[n][nearb]["role"]
                    if cannonical(r):
                        head = graph.node[n]["value"].gold_str()
                        dep = graph.node[nearb]["value"].gold_str()
                        r = graph[n][nearb]["role"]
                        assert isinstance(n,Var),(n,graph.nodes(True))
                        assert isinstance(nearb,Var),(nearb,graph.nodes(True))
                        s.append("# ::edge\t"+head+"\t"+ r+"\t"+dep+"\t"+n._name+"\t"+nearb._name+"\t"+"\n")

        return "".join(s)

