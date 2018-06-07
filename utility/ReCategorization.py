#!/usr/bin/env python3.6
# coding=utf-8
'''

ReCategorizor use a set of templates built from training corpous and deterministic rules
to recombine/recategorize a fragment of AMR graph into a single node for concept identification.
It also stores frequency of wiki for name tuples, and sense for frame concept. (based on training set)

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''
from utility.amr import *
from utility.StringCopyRules import  de_polarity,polarity_match,disMatch
from nltk.metrics.distance import edit_distance

from utility.data_helper import *


import threading
class ReCategorizor(object):

    def template(self):
        # roles lists order in priority, the last role requires alignment
        # [[(r1,c1),(r2,c2)]]
        templates = {}
        templates["play"] = [[(":ARG0", "person"), (":ARG2", "")]]  # concept player
        templates["person"] = [[(":ARG0-of", "")], [(":ARG1-of", "")]]  #
        templates["thing"] = [[(":ARG1-of", "")], [(":ARG0-of", "")],[(":ARG2-of","")]]  #
        templates["most"] = [[(":degree-of", "")]]  #
        templates["-quantity"] = [[(":unit", "")] ]
        templates["monetary-quantity"] = [[(":unit", "")],[(":ARG2-of", "")],[(":ARG1-of", "")], [(":quant", "")]]
        templates["temporal-quantity"] = [[(":quant", "1")],[(":unit", "")],[(":ARG3-of", "")], [(":quant", "1")],[(":unit", "")]]
        templates["date-entity"] = [[(":weekday", "")], [(":dayperiod", "")], [(":season", "")]]
        return templates




    def entity_templates(self):

        def extract_role(myamr, n):
            neabourghs = {}
            cat = myamr.graph.node[n]["value"].le
            aux = None
            for node in myamr.graph[n]:
                neabourghs[myamr.graph[n][node]["role"]] = node

            if ":ARG0" in neabourghs and  myamr.graph.node[neabourghs[":ARG0"]]["value"].le == "person" :
                aux = neabourghs[":ARG0"]

            if ":ARG2" in neabourghs:
                rely_node =  neabourghs[":ARG2"]
                comp = myamr.replace(rely_node,cat,aux,rel=True)
                myamr.rely(n,comp)
                if aux: myamr.rely(aux,comp)
                return True
            elif ":ARG1" in neabourghs:
                rely_node =  neabourghs[":ARG1"]
                comp =  myamr.replace(rely_node,cat,aux,rel=True)
                if aux: myamr.rely(aux,comp)

                return True
            return False

        def extract_named_entity(myamr, n):
            names = []
            match = False
            aux = None
            for node in myamr.graph[n]:
                if myamr.graph.node[node]["value"].cat == Rule_String and ":op" in myamr.graph[n][node]["role"]:
                    match = True
                    names.append(node)  # (role, con,node)
                elif myamr.graph[n][node]["role"] == ":name-of":
                    if self.ner_cat_dict is None or myamr.graph.node[node]["value"].le in self.ner_cat_dict:
                        aux = node
            if match and aux:
                name_head = myamr.replace(names[0],Rule_B_Ner,aux,rel=True)
                myamr.rely(n,name_head)
                myamr.rely(aux,name_head)
                for node in names[1:]:
                    myamr.replace(node,Rule_Ner,aux,rel=True)
            elif match:
                name_head = myamr.replace(names[0],Rule_Ner,rel=True)
                myamr.rely(n,name_head)
                for node in names[1:]:
                    myamr.replace(node,Rule_Ner,rel=True)

            return match


        def extract_other_entity(myamr, n):
            constant_nodes = []
            match = False
            cat = myamr.graph.node[n]["value"].le
            if cat == "rate-entity": return False
            for node in myamr.graph[n]:
                if myamr.graph.node[node]["value"].is_constant():
                    replaced = myamr.replace(node,cat,None,rel=True)
                    if not match:
                        myamr.rely(n,replaced)
                    match = True       #one constant is enough
            return match

        templates = {}
        templates["name"] = extract_named_entity  # use functions to handle the complexity
        templates["-entity"] = extract_other_entity
        templates["-role"] = extract_role
        return templates


    def save(self, path="data/graph_to_node_dict_extended"):
        if self.training :
            graph_counter = self.graph_counter
        else:
            graph_counter = self.old
        graph_to_node_dict_f = Pickle_Helper(path)
        graph_to_node_dict_f.dump(self.senses, "senses")
        graph_to_node_dict_f.dump(self.centers, "centers")
        graph_to_node_dict_f.dump(self.graph_to_node, "graph_to_node")
        graph_to_node_dict_f.dump(self.node_to_graph, "node_to_graph")
        graph_to_node_dict_f.dump(self.wikis, "wikis")
        graph_to_node_dict_f.dump(self.ners, "ners")
        graph_to_node_dict_f.dump(graph_counter, "graph_counter")
        graph_to_node_dict_f.save()
        self.save_to_txt(path+".txt")

    def save_to_txt(self, path="data/graph_to_node_dict.txt",save_counted = False):

        if self.training or save_counted :
            graph_counter = self.graph_counter
        else:
            graph_counter = self.old

        list_dict = []
        for k in self.graph_to_node.keys():
            if k in graph_counter:
                list_dict.append([self.graph_to_node[k].__str__()]
                                 +[val.__str__() for sublist in k for val in sublist]
                                 +[graph_counter[k]])
            else:
                list_dict.append([self.graph_to_node[k].__str__()]
                                 +[val.__str__() for sublist in k for val in sublist]
                                 +[0])
        sorted_counter = sorted(list_dict, key=lambda x: x[-1],reverse=True)

        with open(path, 'w+') as data_file:
            for statistics in sorted_counter:
                string = " ".join(statistics[:-1])+" "+str(statistics[-1])+"\n"
                data_file.write(string)

    def load_from_txt(self, path="data/graph_to_node_dict.txt"):
        def nextLine(f,line):
            tokens = line.split(" ")
            comp = AMRUniversal(string=tokens[0])
            graph = []
            for i in range(1, len(tokens)-1, 2):
                if len(tokens[i + 1].split("(")) < 2:
                #this error is caused by combing numbers in tokenization 400 billion is somehow being mapped to monentary :unit billion ....
                    print(tokens)
                    return f.readline()
                graph.append((tokens[i], AMRUniversal(string=tokens[i + 1])))
            graph = tuple(graph)
            self.update_triple_with_comp(graph[0][1], graph, comp)
            if len(tokens) % 2 == 0:
                self.graph_counter[graph] = int(tokens[-1])
            return f.readline()
        with open(path, 'r') as f:
            line = f.readline()
            while line != "":
                line = nextLine(f,line)


    lock = threading.Lock()
    def __init__(self,from_file=False, path="data/graph_to_node_dict",training=False,auto_convert_threshold = 5,ner_cat_dict = None):
        self.auto_convert_threshold = auto_convert_threshold
        self.old = None
        self.counting = False
        self.converting_txt = None
        self.ner_cat_dict = ner_cat_dict
        self.templates = self.template()
        self.entity_templates = self.entity_templates()
        if from_file:
            graph_to_node_dict_f = Pickle_Helper(path)
            graph_to_node_dict_all = graph_to_node_dict_f.load()
            self.senses = graph_to_node_dict_all["senses"]
            self.ners = graph_to_node_dict_all["ners"]
            self.wikis = graph_to_node_dict_all["wikis"]
           # self.centers = graph_to_node_dict_all["centers"]
          #  self.graph_to_node = graph_to_node_dict_all["graph_to_node"]
          #  self.node_to_graph = graph_to_node_dict_all["node_to_graph"]
          #  self.graph_counter = graph_to_node_dict_all["graph_counter"]
          #  self.sort_graph()
   #         self.ner  =  graph_to_node_dict_all["ner"]
        else:
            self.senses = {}  # record one essential concept in subgraph   c->g could be list
            self.ners = {}
            self.wikis = {}
        self.normalize_prob()
        self.centers = {}  # record one essential concept in subgraph   c->g could be list
        self.graph_to_node = {}  # g -> v
        self.node_to_graph = {}  # v-> g
        self.graph_counter = {}
        self.load_from_txt(path+".txt")
        self.sort_graph()
        if training == True:
            self.training()
        else:
            self.eval()
    def sort_graph(self):
        for node in self.node_to_graph:
            graphs = self.node_to_graph[node]
            self.node_to_graph[node] = sorted(graphs,key = lambda g: self.graph_counter[g],reverse=True)
    Pure_NUM = re.compile(r'[-]?[1-9][0-9]*')

    def unpack_recategorized(self,converted_list,rl,getsense=False,eval= False):

        def unpack_one(uni,ners,index,previous_entity_id):
            def try_fix_frame(uni):
                if (uni.le,uni.cat) in rl.lemmatize_cheat:
                    uni.le = rl.lemmatize_cheat[(uni.le,uni.cat) ]
                    return uni
                else:
                    le  = de_polarity(uni.le)
                    if le and (le,uni.cat) in rl.lemmatize_cheat:
                        uni.le = rl.lemmatize_cheat[(le,uni.cat) ]
                        add_concept(AMRUniversal("-",Rule_Constant,NULL_WORD),1)
                        return uni
                if not eval : return None
                best_dis = .4
                best_lemma = NULL_WORD
                polarity = False
                for con_lemma in rl.frame_lemmas:
                    dis =disMatch(uni.le,con_lemma)
                    if dis < best_dis :
                        best_dis = dis
                        best_lemma = con_lemma
                if best_lemma is not None:
                    uni.le = best_lemma
                    return uni
                uni.le  = de_polarity(uni.le)
                if uni.le:
                    for con_lemma in rl.frame_lemmas:
                        dis =disMatch(uni.le,con_lemma)
                        if dis < best_dis :
                            best_dis = dis
                            best_lemma = con_lemma
                            polarity = True
                    if polarity : add_concept(AMRUniversal("-",Rule_Constant,NULL_WORD),1)
                if best_lemma is not None:
                    uni.le = best_lemma
                    return uni
                return None
            def add_concept(uni,dep=0):
                if len(uni.le) ==0 or uni.le[0] in ["(",")","\""] or uni.le[-1] in ["(",")","\""]:
                    return None
                if uni.cat != Rule_String:
                    for system_str in ["(",")","\"","(",")"]:
                        if system_str in uni.le:
                            uni.cat = Rule_String
                            break

                if uni.cat != Rule_String:
                    uni.le = uni.le.strip("/").strip(":")
                    if ":" in uni.le or "/" in uni.le:
                        uni.cat = Rule_String
                if getsense and uni.cat == Rule_Frame and (uni.sense == "" or uni.sense is None):
                    uni.sense = self.get_senses(uni.le)

                rel_concept.append(uni)
                indexes.append(index)
                dependent_mark.append(dep)
            if PAD_WORD in [uni.cat,uni.le] or\
                NULL_WORD in [uni.cat,uni.le]or\
                UNK_WORD in [uni.cat,uni.le]:
                if len(ners)>0 and previous_entity_id and  previous_entity_id == ners[-1]:
                    return None
                return previous_entity_id

            if uni in self.node_to_graph:
                graph = self.node_to_graph[uni][0]
                top,head = graph[0]
                add_concept(head)
                for r,c in graph[1:]:
                    add_concept(c,1)
             #   print ("composed",uni,out)
                return None

            if uni.cat in ["person","thing"]:
                add_concept(AMRUniversal(uni.cat,Rule_Concept,None))
                uni.cat = Rule_Frame

            if uni.cat == "-":
                polar = AMRUniversal("-",Rule_Constant,None)
                uni.cat = Rule_Frame
                uni = try_fix_frame(uni)
                if uni is None: return None
                if len(rel_concept) > 0 and rel_concept[-1].le != "-":
                    add_concept(polar,1)
                    add_concept(uni)
                else:
                    add_concept(uni)
                    return None

            if uni.cat in [Rule_Ner,Rule_B_Ner]:
                if uni.cat == Rule_B_Ner or  len(ners) == 0  or  ners[-1] != len(rel_concept)-1:
                    if uni.aux and uni.aux not in Special  :
                        entity = AMRUniversal(uni.aux,Rule_Concept,None)
                        add_concept(entity,2)
                        add_concept(AMRUniversal("name",Rule_Concept,None),1)
                    else:
                        add_concept(AMRUniversal("name",Rule_Concept,None))
                add_concept(AMRUniversal(uni.le,Rule_String,None),0)
                entity_id = len(rel_concept)-1
                ners.append(entity_id)
                return entity_id

            if uni.cat == "url-entity":
                entity = AMRUniversal(uni.cat,Rule_Concept,None)
                le = uni.le
                if le.startswith("href=\""):
                    le = le[6:]
                if le.endswith("\">"):
                    le = le[:-2]
                add_concept( AMRUniversal(le,Rule_String,None),1)
                add_concept(entity)
                return len(rel_concept)-1

            if "-entity" in uni.cat:
                if uni.cat in ["rate-entity","rate-entity-3" ]:
                    entity = AMRUniversal("rate-entity",Rule_Frame,"-91")
                    add_concept(AMRUniversal("temporal-quantity",Rule_Concept,NULL_WORD),1)
                    add_concept(AMRUniversal("1",Rule_Num,NULL_WORD),2)
                else:
                    entity = AMRUniversal(uni.cat,Rule_Concept,NULL_WORD)
                if  previous_entity_id is  not None and entity.non_sense_equal(rel_concept[previous_entity_id]):
                    if self.Pure_NUM.search(uni.le) :
                        if len(uni.le) == 6 and uni.cat == "date-entity" :
                            add_concept( AMRUniversal("20"+uni.le[:2],Rule_Num,None))
                            if "00"<uni.le[2:4] < "32":
                                add_concept( AMRUniversal(uni.le[2:4],Rule_Num,None))
                            if "00"<uni.le[4:] < "32":
                                add_concept( AMRUniversal(uni.le[4:],Rule_Num,None))
                        elif ":" in uni.le or "/" in uni.le :
                            add_concept( AMRUniversal(uni.le,Rule_String,None))
                        else:
                            add_concept( AMRUniversal(uni.le,Rule_Num,None))
                    elif uni.le == "-":
                        add_concept( AMRUniversal(uni.le,Rule_Constant,None))
                    else:
                        add_concept( AMRUniversal(uni.le,Rule_Concept,None))
                    return previous_entity_id
                else:
                    if self.Pure_NUM.search(uni.le) :
                        if len(uni.le) == 6 and uni.cat == "date-entity" :
                            add_concept( AMRUniversal("20"+uni.le[:2],Rule_Num,None),1)
                            if "00"<uni.le[2:4] < "32":
                                add_concept( AMRUniversal(uni.le[2:4],Rule_Num,None),1)
                            if "00"<uni.le[4:] < "32":
                                add_concept( AMRUniversal(uni.le[4:],Rule_Num,None),1)
                        elif ":" in uni.le or "/" in uni.le :
                            add_concept( AMRUniversal(uni.le,Rule_String,None),1)
                        else:
                            add_concept( AMRUniversal(uni.le,Rule_Num,None),1)
                    elif uni.le == "-":
                        add_concept( AMRUniversal(uni.le,Rule_Constant,None),1)
                    else:
                        add_concept( AMRUniversal(uni.le,Rule_Concept,None),1)

                    add_concept(entity)
                    return len(rel_concept)-1
            if "-quantity" in uni.cat:
                quantity = AMRUniversal(uni.cat,Rule_Concept,NULL_WORD)
                if self.Pure_NUM.search(uni.le) :
                    add_concept( AMRUniversal(uni.le,Rule_Num,None),1)
                else:
                    add_concept( AMRUniversal(uni.le,Rule_Concept,None),1)
                add_concept(quantity)
                return None

            if "-role" in uni.cat:
                if uni.aux  and uni.aux not in [NULL_WORD,PAD_WORD,UNK_WORD]:
                    add_concept(AMRUniversal(uni.aux,Rule_Concept,NULL_WORD))
                    add_concept(AMRUniversal(uni.cat,Rule_Frame,"-91"),1)
                    add_concept(AMRUniversal(uni.le,Rule_Concept,NULL_WORD),2)
                else:
                    add_concept(AMRUniversal(uni.cat,Rule_Frame,"-91"))
                    add_concept(AMRUniversal(uni.le,Rule_Concept,NULL_WORD),1)
                return None
            uni.aux = NULL_WORD
            if uni.cat != Rule_String:
                uni.le =  uni.le.strip("/")
                uni.le =  uni.le.strip(":")
            if ":" in uni.le: uni.cat = Rule_String
            if uni.cat == Rule_Frame and uni.le not in rl.frame_lemmas:
                uni =  try_fix_frame(uni)
                if uni is None:
                    return None
            if uni.cat == Rule_Concept and "/" in uni.le:
                if uni.le == "1/2": uni.le = "half"
                else: return None
            add_concept(uni)
            return None
        ners = []
        rel_concept = []
        indexes = []
        dependent_mark = []  #marking the level of depedency 0 means primary concept in a recategorized group
        #larger number means high level of dependency. e.g. in Ner_person(Donald) person has 0, name has 1, "Donlad" has 2
        #larger number cannot exist without smaller ones.
        preentity_id = None
        aux_rel = [] #(head_id,dep_id,r)
        assert len(converted_list) > 0,converted_list
        all_is_constant = True
        for i in range(len(converted_list)):
            uni = converted_list[i]
            preentity_id = unpack_one(uni,ners,i,preentity_id)
        for uni in rel_concept:
            if not uni.is_constant():
                all_is_constant = False
                break
        if all_is_constant:
            if len(rel_concept) > 1:
                for type in [Rule_Constant,Rule_String,Rule_Num]:
                    for uni in rel_concept:
                        if uni.cat == type:
                            uni.cat = Rule_Concept
                            all_is_constant = False
                            break
                    if not all_is_constant: break
            else:
                rel_concept = [AMRUniversal("amr-empty",Rule_Concept,None)]
                indexes = [0]
                dependent_mark = [0]
        return rel_concept,indexes,dependent_mark



    def get_senses(self, le):

        if not le in self.senses_probs:
            return "-01"

        probs = self.senses_probs[le] # sense - > role
        return self.most_frequent(probs)  #so far it is worse to use relation feature

    def normalize_prob(self):
        self.senses_probs = {}
        for lemma,counts in self.senses.items():
            self.senses_probs[lemma] = {}
            total = 0 #smoothing
            sense_total = {}
            for sen,count in counts.items():
                sense_total[sen] = 5.0
                total += 5.0
                for nb in count:
                    total += count[nb]
                    sense_total[sen] += count[nb]

            for sen,count in counts.items():
                prob = {}
                prob[None]  = 5.0/sense_total[sen]
                prob["#prior#"]  = sense_total[sen]/total
                prob["#total#"]  = total
                for nb in count:
                    prob[nb] = count[nb]/sense_total[sen]
                self.senses_probs[lemma][sen] = prob


    def most_frequent(self,probs):
        #features: [0/1]*len(counts)-1
        max_y = None
        max_p = 0
        for y,prob in probs.items():
            p = prob["#prior#"]
            if p > max_p:
                max_p = p
                max_y = y
        return max_y


    def read_senses(self, myamr):
        out,rel_out,root_index =  myamr.node_value(all=True)
        for n, roles in rel_out:   #[[self.graph.node[node]["value"],index], [r,index]]
            le, cat, sen = decompose(n[0])
            if cat !=Rule_Frame:continue
            counts =  self.senses.setdefault(le,{})
            counts[sen] = counts.setdefault(sen,{})

            for r, index in roles:
                nb_le = rel_out[index][0][0].le
                counts[sen][nb_le] = counts[sen].setdefault(nb_le,0)+1

    def acc_list(self,key,dict_list):
        for k_i in dict_list:
            if key == k_i[0]:
                k_i[1] += 1
                return
        dict_list.append([key,1])

    def get_wiki(self, names):
        if isinstance(names,list):
            names = tuple(names)
        freqs = self.wikis.setdefault(names,{})
        wiki_f =  AMRUniversal("-",Rule_Constant,None)
        h_f = 0
        for wiki,freq in freqs.items():
            if freq>h_f:
                h_f = freq
                wiki_f =  wiki
        return wiki_f

    def get_ner_type(self, names):
        if isinstance(names,list):
            names = tuple(names)
        freqs = self.ners.setdefault(names,{})
        ner_type_f =  None
        h_f = 0
        for ner_type,freq in freqs.items():
            if freq>h_f:
                h_f = freq
                ner_type_f =  ner_type
        return ner_type_f

    def read_ner(self, myamr):
        ners = myamr.get_ners()  #[ names ,wiki]
        for names,ner_wiki,ner_type in ners:
            for i in range(1,len(names)+1):
                freqs = self.wikis.setdefault(tuple(names[:i]),{})
                freqs[ner_wiki] = freqs.setdefault(ner_wiki,0)+1

                freqs = self.ners.setdefault(tuple(names[:i]),{})
                freqs[ner_type] = freqs.setdefault(ner_type,0)+1


    def update_triple_with_comp(self, center, graph, comp):
        assert center.non_sense_equal(graph[0][1]), "mis match \n" + center + " " + graph

        assert comp.le != "",(center, graph, comp)
        no_sense = AMRUniversal(comp.le, comp.cat, None)
        if graph in self.graph_to_node:
            #   print ("exist",node,graph_to_node[graph],invoker,graph)
            return self.graph_to_node[graph]
        if no_sense in self.node_to_graph:
            return no_sense
        self.graph_to_node[graph] = no_sense
        self.node_to_graph.setdefault(no_sense, []).append(graph)
        dict_center = self.centers.setdefault(center, {})  # search by first element
        dict_center.setdefault(graph[-1], []).append(graph)  # then by last element
        self.graph_counter.setdefault(graph, 0)

        return no_sense

    def update_triple(self, center, graph, le_cat):
        node = AMRUniversal(le_cat[0], le_cat[1], None)
        self.update_triple_with_comp( center, graph, node)
        return node

    def try_entity_template(self, n, key, myamr):
        apply_aux = self.entity_templates[key]
        return apply_aux(myamr, n)

    def counter_up(self, graph):
        self.graph_counter[graph] =self.graph_counter.setdefault(graph,0)+1
        if  self.training and  self.auto_convert_threshold-1 < self.graph_counter[graph] < self.auto_convert_threshold+1:
            print(self.graph_to_node[graph], graph, self.graph_counter[graph])

    def training(self):
        self.training = True
        if self.old is not None:
            self.graph_counter = self.old
            self.old = None
        print ("training mode",self.training)

    def eval(self,t=None):
        self.auto_convert_threshold =  t if t is not None else self.auto_convert_threshold
        self.training = False
        self.old = self.graph_counter
        self.graph_counter = {}
  #      print (self.senses)        print ("testing mode",self.training)
        self.normalize_prob()

    def convert(self, myamr, rl ,snt_token,lemma_token ,pos_token,txt=None):
        self.converting_txt = txt
        for n, v in myamr.node_value():
            le, cat, sen = decompose(v)
            if v.is_constant():
                continue
            for key in self.entity_templates:
                if le.endswith(key) and self.try_entity_template(n, key, myamr):
                    break
        list_con = [(h, r, d) for h, r, d in myamr.triples(normalize_inverses=False,
                                                               normalize_mod=False)
                    if r != ":wiki" and r!=":top" and r != ":instance-of"]
        for i,h_r_d in enumerate(list_con):
            h,r,d = h_r_d
            h, h_v = myamr.var_get_uni(h, True)
            if h_v.is_constant():  # id will be a problem
                continue
            if "rely" not in myamr.graph.node[h] :
                d, d_v = myamr.var_get_uni(d)
                self.convert_one_node_value(h, h_v, r, d_v, myamr,rl,snt_token,lemma_token,pos_token )
        return

    def match(self,myamr, rl ,snt_token,lemma_token ,pos_token,txt=None,jamr=False):
        self.converting_txt = txt
        self.read_senses(myamr)
        self.read_ner(myamr)
        list_con = [(h, r, d) for h, r, d in myamr.triples(normalize_inverses=False,
                                                               normalize_mod=False)
                    if r != ":wiki" and r!=":top"]
        for i,h_r_d in enumerate(list_con):
            h,r,d = h_r_d
            if  r == ":instance-of"  and i+1<len(list_con):
                h,r,d = list_con[i+1]
            h, h_v = myamr.var_get_uni(h, True)
            if h_v.is_constant():  # id will be a problem
                continue
            if "rely" not in myamr.graph.node[h] :
                d, d_v = myamr.var_get_uni(d)
                self.match_one_node_value(h, h_v, r, d_v, myamr,rl,snt_token,lemma_token,pos_token ,jamr)
        return

    def match_one_node_value(self, h, h_v, r, d_v, myamr,rl ,snt_token,lemma_token,pos_token,jamr):
        le, cat, sense = decompose(h_v)
        if (d_v.cat not in Rule_All_Constants or d_v.le != "-")\
                    and not( le=="person" and "people"  in lemma_token):
                if r == ":polarity" :
                    self.polarity_template(myamr, h, h_v, d_v,lemma_token,jamr)
                elif le in self.templates:  # exact match for all others
                    self.try_template(h, h_v, le, myamr,rl,snt_token,lemma_token,pos_token, r, d_v,jamr)
                elif le.endswith("temporal-quantity") :
                    self.try_template(h, h_v, "temporal-quantity", myamr, rl,snt_token,lemma_token,pos_token, r, d_v,jamr)
                elif le.endswith("-quantity") :
                    self.try_template(h, h_v, "-quantity", myamr, rl,snt_token,lemma_token,pos_token, r, d_v,jamr)



    def polarity_template(self, myamr, h, h_v, d_v,lemma_token,jamr):
        le, cat, sense = decompose(h_v)
        if cat != Rule_Frame: return False
        for txt in lemma_token:
            if txt == le: return False
            if polarity_match(txt, le):
                center = h_v
                graph = (("top", h_v), (":polarity", d_v))
                le_cat = (txt, d_v.le)
                self.update_triple(center, graph, le_cat)
                return True
        return False

    # using fixed graph template
    def convert_node(self, center, myamr, graph,rl,snt_token,lemma_token,pos_token):
        result = []
        con_to_role = {}
        for node in myamr.graph[center]:
            if  "has-original" not in myamr.graph.node[node]  :  #avoid using composited concept
                con_to_role[myamr.graph.node[node]["value"]] = (myamr.graph[center][node]["role"], node)  # c->(role, node)
        for r, c in graph:
            if r == ":top":
                result.append(center)
                continue
            if c in con_to_role and r == con_to_role[c][0]:
                result.append(con_to_role[c][1])
            else:
                return False
        universal = self.graph_to_node[graph]
        var = myamr.replace(result[0],universal,rel=True)
        for n in result[1:]:
            myamr.rely(n,var)
        return True

    def try_template(self, n, v, key, myamr, rl,snt_token,lemma_token,pos_token,last_role=None, d_v=None,jamr=False):
        def find_best(candidates,le):
            unique = set(candidates)
            best =candidates[0]
            dis =  edit_distance(best,le)
            for candi in unique:
                dis_i =  edit_distance(candi,le)
                if dis_i < dis:
                    dis = dis_i
                    best = candi
            return best

        role_con_node = []
        if last_role is not None:
            d_le = decompose(d_v)[0]
        for node in myamr.graph[n]:
            if (myamr.graph.node[node]["value"].cat in [Rule_Concept, Rule_Frame] ):
                role_con_node.append(
                    (myamr.graph[n][node]["role"], myamr.graph.node[node]["value"], node))  # (role, con,node)
                # [t for t in self.templates[key] if t[-1][0] == last_role and t[-1][1] in d_le]
        for template in sorted(self.templates[key],key=len,reverse=True):
            if last_role is not None and not (template[-1][0] == last_role and template[-1][1] in d_le):
                continue
            filled = [(":top", v, n)]
            matched = 0 if len(template) > 1 else 1
            for i, r_c in enumerate(template):
                r, c = r_c
                matched = 0
                for ri, ci, ni in role_con_node:
                    if r == ri and ci.le.endswith(c):
                        filled.append((ri, ci, ni))  # arbitaryly assign to the first match
                        matched += 1
                if not matched == 1:
                    break
            if not matched == 1: continue

            if filled[-1][-1] not in myamr.graph.node:
                return False
            if jamr and myamr.graph.node[filled[-1][-1]]["align"] is not None:
                align_txts = [lemma_token[myamr.graph.node[filled[-1][-1]]["align"]]]
            else:
                align = rl.match_concept(snt_token,myamr.graph.node[filled[-1][-1]]["value"],lemma_token,pos_token,with_target = True)
                align_txts = [i_t_p[1]   for  i_t_p in align ]  #if i_t_p[2] in ["NOUN","NUM",'PROPN',"NN","CD","PRP$"]
      #      if key == "-quantity":print ("key1\n",v,role_con_node,myamr)
            if len(align_txts) == 0 :return False
            align_txt = find_best(align_txts,myamr.graph.node[filled[-1][-1]]["value"].le)
            center = filled[0][1]
            le_cat = (align_txt, decompose(center)[0])  # aribitaryly pick the first aligned txt
            graph = tuple([(ri, ci) for ri, ci, ni in filled])
      #      if key == "-quantity":print (v,role_con_node,graph,myamr)
            self.update_triple(center, graph, le_cat)
            return True  # return since the order decides priority
        return False

    def convert_one_node_value(self, h, h_v, r, d_v, myamr,rl ,snt_token,lemma_token,pos_token):
        if h_v in self.centers and (r, d_v) in self.centers[h_v]:
            for subgraph in sorted(self.centers[h_v][(r, d_v)],key=len,reverse=True):
                if self.convert_node( h, myamr, subgraph,rl,snt_token,lemma_token,pos_token):
                    self.counter_up(subgraph)
                    return True
        return False
