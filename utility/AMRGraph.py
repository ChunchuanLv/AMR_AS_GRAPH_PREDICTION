#!/usr/bin/env python3.6
# coding=utf-8
'''

AMRGraph builds on top of AMR from amr.py
representing AMR graph as graph,
and extract named entity (t1,..,tn, ner type, wiki) tuple. (we use model predicting for deciding ner type though)
Being able to apply recategorization to original graph,
which involves collapsing nodes for concept identification and unpacking for relation identification.

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-28
'''
from utility.amr import *
from utility.constants import *
import networkx as nx

class AMRGraph(AMR):
    def __init__(self, anno, normalize_inverses=True,
                 normalize_mod=False, tokens=None,aligns={}):
        '''
        create AMR from text, and convert AMR to AMRGraph of standard representation
        '''
        super().__init__(anno, tokens)
        self.ners = []
        self.gold_concept = []
        self.gold_triple = []
        self.graph = nx.DiGraph()
        self.wikis = []
        for h, r, d in [(h, r, d) for h, r, d in self.triples(normalize_inverses=normalize_inverses,
                                                              normalize_mod=normalize_mod) if
                        (r != ":instance" )]:
            if r == ':wiki':
                h, h_v = self.var_get_uni(h, True,(h, r, d ))
                d, d_v = self.var_get_uni(d)
                self.wikis.append(d)
                self.ners.append((h,d_v))
                continue
            elif r == ':top':
                d, d_v = self.var_get_uni(d)
                self.root = d
                self.graph.add_node(d, value=d_v, align=None,gold=True)
            else:
                h, h_v = self.var_get_uni(h, True,(h, r, d ))
                d, d_v = self.var_get_uni(d)
                self.graph.add_node(h, value=h_v, align=None,gold=True)
                self.graph.add_node(d, value=d_v, align=None,gold=True)
                self.graph.add_edge(h, d, role=r)
                self.graph.add_edge(d, h, role=r + "-of")

                # for i in self._triples:
                #     print(i)
        self.read_align(aligns)

    #alignment from copying mechanism
    def read_align(self, aligns):
        for prefix in aligns:
            i = self._index[prefix]
            if isinstance(i,Var):
                assert i  in self.graph.node,(self.graph.nodes(True),self.triples(normalize_inverses=True,
                                                              normalize_mod=False),self._anno)
                self.graph.node[i]["align"] = aligns[prefix]
            else:
                if Var(prefix) in self.wikis: continue
                assert Var(prefix)  in self.graph.node,(prefix,aligns,self._index,self.graph.nodes(True),self._anno)
                self.graph.node[Var(prefix)]["align"] = aligns[prefix]


    def check_consistency(self,pre2c):
        for prefix in pre2c:
            var = self._index[prefix]
            if not isinstance(var,Var): var = Var(prefix)
            if var in self.wikis: continue
            assert var  in self.graph.node,(prefix, "\n",pre2c,"\n",self.graph.node,"\n",self._anno)
            amr_c = self.graph.node[var]["value"]

            assert amr_c.gold_str() == pre2c[prefix],(prefix, var,amr_c.gold_str() ,pre2c[prefix],"\n",pre2c,"\n",self.graph.nodes(True))

    def get_gold(self):
        cons = []
        roles = []
        for n, d in self.graph.nodes(True):
            if "gold" in d:
                v = d["value"]
                cons.append(v)

        for h, d, rel in self.graph.edges(data=True):
            r = rel["role"]
            if self.cannonical(r):
                assert "gold" in self.graph.node[h] and "gold" in self.graph.node[d]
                h = self.graph.node[h]["value"]
                d = self.graph.node[d]["value"]
                roles.append([h,d,r])

        root = self.graph.node[self.root]["value"]
        roles.append([AMRUniversal(BOS_WORD,BOS_WORD,NULL_WORD),root,':top'])
        return cons,roles

    def get_ners(self):
        ners = []
        for v,wiki in self.ners:  #v is name variable
            name = None
            names = []
            for nearb in self.graph[v]:
                if self.graph[v][nearb]["role"] == ":name":
                    name = nearb
                    break
            if name is None:
                print  (self.graph[v],self._anno)
                continue
            ner_type = self.graph.node[v]["value"]
            for node in self.graph[name]:
                if self.graph.node[node]["value"].cat == Rule_String and ":op" in self.graph[name][node]["role"]:
                    names.append(( self.graph.node[node]["value"],int(self.graph[name][node]["role"][-1])))  # (role, con,node)

            names = [t[0] for t in sorted(names,key = lambda t: t[1])]
            ners.append([names,wiki,ner_type])
        return ners



    def rely(self,o_node,n_node):
        if "rely" in  self.graph.node[o_node]:
            return
        self.graph.node[o_node].setdefault("rely",n_node)

    def link(self,o_node,n_node,rel):
        self.graph.node[o_node].setdefault("original-of",[]).append( n_node ) # for storing order of replacement
        if n_node:
            self.graph.node[n_node]["has-original"] = o_node  # for storing order of replacement
            self.graph.node[n_node]["align"] = self.graph.node[o_node]["align"]
            if rel: self.rely(o_node,n_node)

    def replace(self,node,cat_or_uni,aux=None,rel=False):

        aux_le =  self.graph.node[aux]['value'].le if aux else None

        if isinstance(cat_or_uni,AMRUniversal):
            universal = cat_or_uni
        else:
            le = self.graph.node[node]['value'].le
            universal = AMRUniversal(le, cat_or_uni, None, aux_le)  #aux_le is usually named entity type
        # create a new recategorized node
        # gold is not marked, so new recategorized node won't be used for relation identification
        var = Var(node._name+"_"+universal.cat)
        self.graph.add_node(var, value=universal, align=None)
        self.link(node,var,rel)

        return var


    #get a amr universal node from a variable in AMR or a constant in AMR
    def var_get_uni(self, a, head=False,tri=None):
        if isinstance(a,Var):
            return a, AMRUniversal(concept=self._v2c[a])
        else:
            if head:
                assert False, "constant as head" + "\n" + a + self._anno+"\n"+str(tri)
            return Var(a), AMRUniversal(concept=self._index[a])



    def __getitem__(self, item):
        return self.graph.node[item]

    #check whether the relation is in the cannonical direction
    def cannonical(self,r):
        return  "-of" in r and not self.is_core(r) or  "-of"  not in r and  self.is_core(r)

    def getRoles(self,node,index_dict,rel_index,relyed = None):
        # (amr,index,[[role,rel_index]])
        if relyed and relyed not in index_dict:
            print ("rely",node,relyed,self.graph.node[relyed]["value"],index_dict,self._anno)
        elif relyed is None and node not in index_dict: print (self.graph.node[node]["value"])
        index = index_dict[node] if relyed is None else index_dict[relyed]
        out = []
     #   if self.graph.node[node]["value"].le != "name":
        for n2 in  self.graph[node]:
            r = self.graph[node][n2]["role"]
            if self.cannonical(r):
                if n2 not in rel_index:
                    print(node,n2)
                    print(self._anno)
                out.append([r,rel_index[n2]])
        return [[self.graph.node[node]["value"],index], out]

    #return data for training concept identification or relation identification
    def node_value(self, keys=["value"], all=False):
        def concept_concept():
            out = []
            index = 0
            index_dict ={}
            for n, d in self.graph.nodes(True):
                if "original-of"in d:
                    comps = d["original-of"]
                    for comp in comps:
                        if comp is None:
                            continue
                        comp_d = self.graph.node[comp]
                        out.append([comp] + [comp_d[k] for k in keys])
                        index_dict[comp] = index
                        index += 1
                elif not ("has-original" in d or  "rely" in d):
                    out.append([n] + [d[k] for k in keys])
                    index_dict[n] = index
                    index += 1
            return out,index_dict
        def rel_concept():
            index = 0
            rel_index ={}
            rel_out = []
            for n, d in self.graph.nodes(True):
                if "gold" in d:
                    rel_out.append([n,d])
                    rel_index[n] = index
                    index += 1

            return rel_out,rel_index

        out,index_dict = concept_concept()
        if all:
            rel_out, rel_index = rel_concept()
            for i, n_d in enumerate( rel_out):
                n,d = n_d
                if "rely" in d:
                    rel_out[i] =self.getRoles(n,index_dict,rel_index,d["rely"])
                elif not ("has-original" in d or  "original-of" in d):
                    rel_out[i] = self.getRoles(n,index_dict,rel_index)
                else:
                    assert False , (self._anno,n,d["value"])
            assert (self.root  in rel_index),(self.graph.nodes[self.root],rel_index,self._anno)
            return out,rel_out,rel_index[self.root]
        else:
            return out