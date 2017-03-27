#!/usr/bin/env python2.7
# coding=utf-8
'''
Parser for Abstract Meaning Represention (AMR) annotations in Penman format.
A *parsing expression grammar* (PEG) for AMRs is specified in amr.peg
and the AST is built by the Parsimonious library (https://github.com/erikrose/parsimonious).
The resulting graph is represented with the AMR class.
When called directly, this script runs some cursory unit tests.

If the AMR has ISI-style inline alignments, those are stored in the AMR object as well.

TODO: Include the smatch evaluation code
(released at http://amr.isi.edu/evaluation.html under the MIT License).

@author: Nathan Schneider (nschneid@inf.ed.ac.uk)
@since: 2015-05-05
'''
from utility.amr import *
import networkx as nx


class myAMR(AMR):
    
    def __init__(self, anno, tokens=None):
        '''
        Given a Penman annotation string for a single rooted AMR, construct the data structure.
        Triples are stored internally in an order that preserves the layout of the
        annotation (even though this doesn't matter for the "pure" graph).
        (Whitespace is normalized, however.)

        Will raise an AMRSyntaxError if notationally malformed, or an AMRError if
        there is not a 1-to-1 mapping between (unique) variables and concepts.
        Will not check details such as the appropriateness of relation/role names
        or constants. Does not currently read or store metadata about the AMR.
        '''
        super().__init__(anno,tokens)
        self.graph = nx.Graph()
        for h, r, d in [(h, r, d) for h, r, d in self.triples()
              if r not in (':instance', ':instance-of')]:
            if r == ':top':
                self.root = d
            else:
                self.graph.add_node(h)
                self.graph.add_node(d)
                self.graph.add_edge(h,d,label = r)

        # for i in self._triples:
        #     print(i)
    def n_to_c(self,n):
        if n.is_var():
            return self._v2c[n]
        else:
            return n
    def bfs(self):
        return [(self.n_to_c(e[0]),self.graph[e[0]][e[1]]['label'],self.n_to_c(e[1])) for e in nx.bfs_edges(self.graph,self.root)]
        
