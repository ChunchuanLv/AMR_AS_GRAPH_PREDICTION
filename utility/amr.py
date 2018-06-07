#!/usr/bin/env python3.6
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


@author:Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-28
'''

from collections import defaultdict
from collections import Counter
from parsimonious.grammar import Grammar
from nltk.parse import DependencyGraph
from utility.constants import *

PUSH = "<STACK_PUSH>"
POP = "<STACK_POP>"
KEEP = "<STACK_KEEP>"


def clean_grammar_file(s):
    return re.sub(
        '\n[ \t]+',
        ' ',
        re.sub(
            r'#.*',
            '',
            s.replace(
                '\t',
                ' ').replace(
                '`',
                '_backtick')))

with open(os.path.join(os.path.dirname(__file__), 'amr.peg')) as inF:
    grammar = Grammar(clean_grammar_file(inF.read()))

class AmrObject(object):
    
    def __init__(self, name):
        self._name = name
        
    def is_var(self):
        return True
    
    def is_constant(self):
        return False

    def deepcopy(self,s=""):
        return AmrObject(self._name+s)




NULL_WORD = ""

#Data structure representing AMR nodes
class AMRUniversal(object):


    def __init__(self, *args, **kwargs):
        if "string" in kwargs:
            self.construct_by_string(kwargs["string"])
        elif "concept" in kwargs:
            self.construct_by_concept(kwargs["concept"])
        else:
            aux = args[3] if len(args)>3 else NULL_WORD
            self.construct_by_content(args[0],args[1],args[2],aux)

    def construct_by_string(self, txt):
        txts = txt[:-1].split("(")
        assert len(txts)>1,txt
        assert len(txts[1])>0 and len(txts[0])>0,txt
        while  txts[1][-1] == ")" :
            txts[1] = txts[1][:-1]
        cats = txts[0].split("_")

        self.cat = cats[0]
        self.aux = cats[1] if len(cats)>1 else NULL_WORD
        self.le = re.sub(RE_FRAME_NUM,"", txts[1])
        self.sense = re.search(RE_FRAME_NUM,  txts[1]).group() \
            if re.search(RE_FRAME_NUM,  txts[1]) else NULL_WORD

    def is_concept(self):

        return self.cat == Rule_Concept

    def is_var(self):
        return False

    def is_constant(self):
        return self.cat in Rule_All_Constants

    def is_frame(self):
        return self.cat == Rule_Frame

    def construct_by_content(self, le,cat,sense,aux):
        assert isinstance(le, str) and isinstance(cat, str),(le,cat,sense,aux)
        self.le = le
        self.cat = cat
        self.sense = sense if sense is not None else NULL_WORD
        self.aux = aux if aux is not None else NULL_WORD

    def construct_by_concept(self, concept):
        le,cat,sense = decompose(concept)
        self.le = le
        self.cat = cat
        self.sense = sense if sense is not None else NULL_WORD
        self.aux = NULL_WORD

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        sense = self.sense if self.sense != NULL_WORD else ""
        return self.cat+"("+self.le+sense+")" if self.aux is  NULL_WORD \
            else self.cat+"_"+self.aux+"("+self.le+sense+")"

    def gold_str(self):
        sense = self.sense if self.sense != NULL_WORD else ""
        if self.cat == Rule_String: return "\""+self.le+"\""
        return self.le+sense

    def __hash__(self):
        return hash(self.__repr__())


    def non_sense_equal(self,other):
        return isinstance(other,AMRUniversal) and self.le == other.le and self.cat ==other.cat
    def __eq__(self, other):
        return self.le == other.le and self.cat ==other.cat \
               and self.sense ==other.sense and self.aux ==other.aux

#decompose amr node to le_cat_sense
def decompose(c):

    if c is None: return None,None,None
    if isinstance(c,AMRUniversal):
        return c.le,c.cat,c.sense

    if c.is_var():
        return c._value,Rule_Re,None
    if c.is_constant():
        if c.is_str():
            return c._value,Rule_String,None
        if c.is_num():
            return c._value,Rule_Num,None
        return c._value,Rule_Constant,None

    x = c.__str__().split(Splish)
    if len(x)>1 :
        sense = re.search(RE_FRAME_NUM, x[0]).group()
        return re.sub(RE_FRAME_NUM,"",x[0]),x[1],sense

    if c.is_frame():
        return re.sub(RE_FRAME_NUM,"",c._name),Rule_Frame,c._name[-3:]
    return c.__str__(),Rule_Concept,None

class Var(object):

    def __init__(self, name,align = ""):
        self._name = name
        self._align = align
    
    def __le__(self,other):
        return self._name < other._name
    def set_align(self,align):
        self._align = align
    def __align__(self):
        return self._align[1:]
    def is_var(self):
        return True
    def is_concept(self):
        return False
    def is_constant(self):
        return False

    def __repr__(self):
        return 'Var(' + self._name + self._align+')'

    def __str__(self):
        return self._name

    # args are ignored, but this is present so Var objects behave like objects
    # that can have alignments
    def __call__(self, **kwargs):
        return self.__str__()

    def __eq__(self, that):
        return isinstance(that, type(self)) and self._name == that._name

    def __hash__(self):
        return hash(self._name)

    def deepcopy(self,s=""):
        return Var(self._name+s,self._align)


class Concept(object):
    RE_FRAME_NUM = re.compile(r'-\d\d$')
    COMP = re.compile(r'_\d')
    def __init__(self, name,align = None):
        self._name = name
    
    def is_concept(self):

        return True
    def is_var(self):
        return False
    def is_constant(self):
        return False
    def is_frame(self):
        return self.RE_FRAME_NUM.search(self._name) is not None
    def is_compo(self):
        return  self.COMP.search(self._name) is not None

    def __repr__(self):
        return 'Concept(' +self._name.replace(Splish,"_") + ')'

    def __str__(self, align={}):
        return self._name + align.get(self, '')

    def __call__(self, **kwargs):
        return self.__str__(**kwargs)

    def __eq__(self, that):
        return isinstance(that, type(self)) and self._name == that._name

    def __hash__(self):
        return hash(repr(self))
    def deepcopy(self,s=""):
        return Concept(self._name+s)
    


class AMRConstant(object):

    def __init__(self, value):
        self._value = value

    def is_concept(self):
        return False
    def is_var(self):
        return False
    def is_constant(self):
        return True
    
    def is_num(self):
        return False
    
    def is_str(self):
        return False

    def is_frame(self):
        return False

    def __repr__(self):
        return 'Const(' + self._value + ')'

    def __str__(self, align={}):
        return self._value + align.get(self, '')

    def __call__(self, **kwargs):
        return self.__str__(**kwargs)

    def __eq__(self, that):
        return isinstance(that, type(self)) and self._value == that._value

    def __hash__(self):
        return hash(repr(self))
    def deepcopy(self,s=""):
        return AMRConstant(self._value+s)


class AMRString(AMRConstant):


    
    def __repr__(self):
        return '"' + self._value + '"'
    
    def is_str(self):
        return True

    def deepcopy(self,s=""):
        return AMRString(self._value+s)

class AMRNumber(AMRConstant):

    def __repr__(self):
        return 'Num(' + self._value + ')'
    
    
    def is_num(self):
        return  self._value != "-"
    def deepcopy(self,s=""):
        return AMRNumber(self._value+s)


class AMRError(Exception):
    pass


class AMRSyntaxError(Exception):
    pass

#used to check whether its' the cannonical form of relation
#only core relation and the inverse of non-core relation is in cannonical form
def is_core(r):
    return r.startswith(':ARG') or  r.startswith(':op') or  r.startswith(':snt') or r.startswith(':top')

class AMR(DependencyGraph):
    '''
    An AMR annotation. Constructor parses the Penman notation.
    Does not currently provide functionality for manipulating the AMR structure,
    but subclassing from DependencyGraph does provide the contains_cycle() method.

    >>> s = """                                    \
    (b / business :polarity -                      \
       :ARG1-of (r / resemble-01                   \
                   :ARG2 (b2 / business            \
                             :mod (s / show-04)))) \
    """
    >>> a = AMR(s)
    >>> a
    (b / business :polarity -
        :ARG1-of (r / resemble-01
            :ARG2 (b2 / business
                :mod (s / show-04))))
    >>> a.reentrancies()
    Counter()
    >>> a.contains_cycle()
    False

    >>> a = AMR("(h / hug-01 :ARG0 (y / you) :ARG1 y :mode imperative)")
    >>> a
    (h / hug-01
        :ARG0 (y / you)
        :ARG1 y
        :mode imperative)
    >>> a.reentrancies()
    Counter({Var(y): 1})
    >>> a.contains_cycle()
    False

    >>> a = AMR('(h / hug-01 :ARG1 (p / person :ARG0-of h))')
    >>> a
    (h / hug-01
        :ARG1 (p / person
            :ARG0-of h))
    >>> a.reentrancies()
    Counter({Var(h): 1})
    >>> a.triples()     #doctest:+NORMALIZE_WHITESPACE
    [(Var(TOP), ':top', Var(h)), (Var(h), ':instance-of', Concept(hug-01)),
     (Var(h), ':ARG1', Var(p)), (Var(p), ':instance-of', Concept(person)),
     (Var(p), ':ARG0-of', Var(h))]
    >>> a.contains_cycle()
    [Var(p), Var(h)]

    >>> a = AMR('(h / hug-01 :ARG0 (y / you) :mode imperative \
    :ARG1 (p / person :ARG0-of (w / want-01 :ARG1 h)))')
    >>> # Hug someone who wants you to!
    >>> a.contains_cycle()
    [Var(w), Var(h), Var(p)]

    >>> a = AMR('(w / wizard    \
    :name (n / name :op1 "Albus" :op2 "Percival" :op3 "Wulfric" :op4 "Brian" :op5 "Dumbledore"))')
    >>> a
    (w / wizard
        :name (n / name
            :op1 "Albus"
            :op2 "Percival"
            :op3 "Wulfric"
            :op4 "Brian"
            :op5 "Dumbledore"))

    # with automatic alignments
    # at_0 a_1 glance_2 i_3 can_4 distinguish_5 china_6 from_7 arizona_8 ._9
    >>> a = AMR('(p / possible~e.4 :domain~e.1 (d / distinguish-01~e.5 :arg0 (i / i~e.3) \
    :arg1 (c / country :wiki~e.7 "china"~e.6 :name (n / name :op1 "china"~e.6))          \
    :arg2 (s / state :wiki~e.7 "arizona"~e.8 :name (n2 / name :op1 "arizona"~e.8))       \
    :manner~e.0 (g / glance-01~e.2 :arg0 i)))')
    >>> a
    (p / possible~e.4
        :domain~e.1 (d / distinguish-01~e.5
            :arg0 (i / i~e.3)
            :arg1 (c / country
                :wiki~e.7 "china"~e.6
                :name (n / name
                    :op1 "china"~e.6))
            :arg2 (s / state
                :wiki~e.7 "arizona"~e.8
                :name (n2 / name
                    :op1 "arizona"~e.8))
            :manner~e.0 (g / glance-01~e.2
                :arg0 i)))
    >>> a.alignments()  #doctest:+NORMALIZE_WHITESPACE
    {(Var(d), ':manner', Var(g)): 'e.0', Concept(glance-01): 'e.2',
    "china": 'e.6', (Var(s), ':wiki', "arizona"): 'e.7', Concept(i): 'e.3',
    "arizona": 'e.8', (Var(p), ':domain', Var(d)): 'e.1',
    Concept(distinguish-01): 'e.5', Concept(possible): 'e.4',
    (Var(c), ':wiki', "china"): 'e.7'}
    >>> print(a(alignments=False))
    (p / possible
        :domain (d / distinguish-01
            :arg0 (i / i)
            :arg1 (c / country
                :wiki "china"
                :name (n / name
                    :op1 "china"))
            :arg2 (s / state
                :wiki "arizona"
                :name (n2 / name
                    :op1 "arizona"))
            :manner (g / glance-01
                :arg0 i)))
    '''

    @classmethod
    def triples2String(self, triples):
        a = AMR("(a / a)")
        a._triples = []
        a._constants = set()
        a._v2c = {}
        a._alignments = {}
        variables = {}
        constants = {}
        concepts = {}

        # variables[triples[0][0]] = Var(triples[0][0])
        # concepts[triples[0][1]] = Concept(triples[0][1])
        #a._triples.append((Var('TOP'), ':top', variables[triples[0][0]]))
        #a._triples.append((variables[triples[0][0]],':instance-of', concepts[triples[0][1]]))
        variables["TOP"] = Var("TOP")
        concepts["TOP"] = Concept("TOP")
        variables[triples[0][3]] = Var(triples[0][3])
        concepts[triples[0][4]] = Concept(triples[0][4])
        a._triples.append((Var('TOP'), ':top', variables[triples[0][3]]))
        # a._triples.append((variables[triples[0][3]],':instance-of',concepts[triples[0][4]]))
        old_instances = []
        for i in range(0, len(triples)):
            (v1, c1, label, v2, c2) = triples[i]
            # if label == ":top":
            #     continue
            instance_triple = ""
            if c1 == "" and v1 != "TOP":
                if v1 not in constants:
                    constants[v1] = AMRConstant(v1)

                a._constants.add(constants[v1])
                t1 = constants[v1]
            else:
                if v1 not in variables:
                    variables[v1] = Var(v1)
                if c1 not in concepts:
                    concepts[c1] = Concept(c1)

                a._v2c[variables[v1]] = concepts[c1]
                t1 = variables[v1]

            if c2 == "" and v2 != "TOP":
                if v2 not in constants:
                    constants[v2] = AMRConstant(v2)

                a._constants.add(constants[v2])
                t2 = constants[v2]
            else:
                if v2 not in variables:
                    variables[v2] = Var(v2)
                if c2 not in concepts:
                    concepts[c2] = Concept(c2)

                usedrightafter = ((i < len(triples) - 1)
                                  and triples[i + 1][0] == v2)
                noothermentions = (
                    len([t for t in triples[i + 1:] if t[0] == v2 or t[3] == v2]) == 0)
                # print(v2)
                # print([t for t in triples[i+1:] if t[0] == v2 or t[3] == v2])
                # print(usedrightafter, noothermentions)
                if v2 not in old_instances and (
                        noothermentions or usedrightafter):
                    instance_triple = (
                        variables[v2], ':instance-of', concepts[c2])
                    old_instances.append(v2)
                a._v2c[variables[v2]] = concepts[c2]
                t2 = variables[v2]

            if label != ":top":
                a._triples.append((t1, label, t2))
            if instance_triple != "":
                a._triples.append(instance_triple)

        #a._triples = [(Var("TOP"), ':top', Var("c")), (Var("c"), ':instance-of', Concept("chapter")), (Var("c"), ':mod', Var("7")), (Var("7"), ':instance-of', Concept("7"))]
        #print (a._triples)
        return a

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
        self._v2c = {}
        self.v2prefix = {}
        self._triples = []
        self._constants = set()
        self._alignments = {}
        self._tokens = tokens
        self.nodes = defaultdict(lambda: {'address': None,
                                          'type': None,
                                          'head': None,
                                          'rel': None,
                                          'word': None,
                                          'deps': []})
        # Emulate the DependencyGraph (superclass) data structures somewhat.
        # There are some differences, e.g., in AMR it is possible for a node to have
        # multiple dependents with the same relation; so here, 'deps' is simply a list
        # of dependents, not a mapping from relation types to dependents.
        # In typical depenency graphs, 'word' is a word in the sentence
        # and 'address' is its index; here, both point to the object representing
        # the node's AMR variable, concept, or constant.

        TOP = Var('TOP')
        self.nodes[TOP]['address'] = self.nodes[TOP]['word'] = TOP
        self.nodes[TOP]['type'] = 'TOP'
        if anno:
            self._anno = anno
            p = grammar.parse(anno)
            if p is None:
                raise AMRSyntaxError(
                    'Well-formedness error in annotation:\n' + anno.strip())
            self._analyze(p)



    def _analyze(self, p):

        '''Analyze the AST produced by parsimonious.'''
        v2c = {}    # variable -> concept
        elts = {}  # for interning variables, concepts, constants, etc.
        consts = set()  # all constants used in the A
        index = {}  #var to index
        align_var = []

        def intern_elt(x,prefix):
            index[prefix] =  x
            if isinstance(x,Var):
                return elts.setdefault(x,x)

            return elts.setdefault(prefix,prefix)
        def walk(n,prefix = "0."):    # (v / concept...)
            triples = []
            deps = []
            v = None
            i = 0
            for ch in n.children:
                t = ch.expr_name
                if t == 'VAR':
                    var_node, alignment_node = ch.children
                    x = Var(var_node.text,alignment_node.text)
                    v = intern_elt(x,".".join(prefix.split(".")[:-1]))
          #          print ("saw var",v,prefix)
                elif t == 'CONCEPT' :

                    assert v is not None
                    concept_node, alignment_node = ch.children
                    if  v in v2c:
                        raise AMRError(
                            'Variable has multiple concepts: ' +
                            str(v2c)+
                            '\n' +
                            concept_node.text +
                            '\n' +
                            str(v) +
                            '\n' +
                            str(v2c[v])  +
                            '\n' +
                            self._anno)
                    v2c[v] = Concept(concept_node.text)
                elif t == '' and ch.children:
                    old_i = i
                    for ch2 in ch.children:
                        _part, RELpart, _part, Ypart = ch2.children
                        rel, relalignment = RELpart.children
                        rel = rel.text
                        assert rel is not None
                        assert len(Ypart.children) == 1
                        q = Ypart.children[0]
                        tq = q.expr_name
                        n2 = None
                        triples2 = []
                        deps2 = []
                        if tq == 'X':
                            n2, triples2, deps2 = walk(q,prefix+str(i)+".")
                            i += 1
                        else:
                            if tq == 'NAMEDCONST':
                                qleft, qalign = q.children
                                n2 = intern_elt(AMRConstant(qleft.text),prefix+str(i))
                                consts.add(n2)
                                i += 1
                            elif tq == 'VAR':
                                qleft, qalign = q, None
                                   
                                var_node, alignment_node = qleft.children
                                n2 = intern_elt(Var(var_node.text,alignment_node.text),prefix+str(i))
                            elif tq == 'STR':
                                quote1, qstr, quote2, qalign = q.children
                                n2 = intern_elt(AMRString(qstr.text),prefix+str(i))
                                consts.add(n2)
                                i += 1
                            elif tq == 'NUM':
                                qleft, qalign = q.children
                                n2 = intern_elt(AMRNumber(qleft.text),prefix+str(i))
                                consts.add(n2)
                                i += 1
                            if qalign and len(qalign.text) > 0:
                                self._alignments[n2] = qalign.text[1:]
                        assert n2 is not None
                        self.add_node({'address': n2, 'word': n2, 'type': tq,
                                       'rel': rel, 'head': v})
                        self.nodes[n2]['deps'].extend(deps2)
                        deps.append(n2)
                        triples.append((v, rel, n2))
                        triples.extend(triples2)
                    i = old_i

            return v, triples, deps

        assert p.expr_name == 'ALL'
        n = None
        for ch in p.children:
            if ch.expr_name == 'X':
                assert n is None    # only one top-level node per AMR
                n, triples, deps = walk(ch)
                self.nodes[n]['deps'].extend(deps)
                triples = [(Var('TOP'), ':top', n)] + triples


        # All is well, so store the resulting data
        self._v2c = v2c
        self._triples = triples
        self._constants = consts
        self._index = index

    # overrides superclass implementation
    def triples(
            self,
            head=None,
            rel=None,
            dep=None,
            normalize_inverses=False,
            normalize_mod=False):
        '''
        Returns a list of head-relation-dependent triples in the AMR.
        Can be filtered by specifying a value (or iterable of allowed values) for:
          - 'head': head variable(s)
          - 'rel': relation label(s) (string(s) starting with ":"), or "core" for all :ARGx roles,
            or "non-core" for all other relations. See also role_triples().
          - 'dep': dependent variable(s)/concept(s)/constant(s)
        Boolean options:
          - 'normalize_inverses': transform (h,':REL-of',d) relations to (d,':REL',h)
          - 'normalize_mod': transform ':mod' to ':domain-of' (before normalizing inverses,
            if applicable)

        >>> a = AMR('(h / hug-01 :ARG1 (p / person :ARG0-of h))')
        >>> a.triples(head=Var('h'))
        [(Var(h), ':instance-of', Concept(hug-01)), (Var(h), ':ARG1', Var(p))]
        >>> a.triples(head=Var('p'), rel=':instance-of')
        [(Var(p), ':instance-of', Concept(person))]
        >>> a.triples(rel=[':top',':instance-of'])
        [(Var(TOP), ':top', Var(h)), (Var(h), ':instance-of', Concept(hug-01)), (Var(p), ':instance-of', Concept(person))]
        >>> a.triples(rel='core')
        [(Var(h), ':ARG1', Var(p)), (Var(p), ':ARG0-of', Var(h))]
        >>> a.triples(rel='core', normalize_inverses=True)
        [(Var(h), ':ARG1', Var(p)), (Var(h), ':ARG0', Var(p))]
        '''
        tt = (trip for trip in self._triples)
        if normalize_mod:
            tt = ((h, ':domain-of', d) if r == ':mod' else (h, r, d)
                  for h, r, d in tt)
        if normalize_inverses:
            tt = ((y, r[:-3], x) if r.endswith('-of') else (x, r, y)
                  for x, r, y in tt)
        if head:
            tt = (
                (h, r, d) for h, r, d in tt if h in (
                    head if hasattr(
                        head, '__iter__') else (
                        head,)))
        if rel:
            if rel == 'core':
                tt = ((h, r, d) for h, r, d in tt if r.startswith(':ARG'))
            elif rel == 'non-core':
                tt = ((h, r, d) for h, r, d in tt if not r.startswith(':ARG'))
            else:
                tt = (
                    (h, r, d) for h, r, d in tt if r in (
                        rel if hasattr(
                            rel, '__iter__') else (rel)))
        if dep:
            tt = (
                (h, r, d) for h, r, d in tt if d in (
                    dep if hasattr(
                        dep, '__iter__') else (
                        dep,)))
        return list(tt)

    def is_core(self,r):
            return is_core(r)
    def role_triples(self, **kwargs):
        '''
        Same as triples(), but limited to roles (excludes :instance-of, :instance, and :top relations).

        >>> a = AMR('(h / hug-01 :ARG1 (p / person :ARG0-of h))')
        >>> a.role_triples()
        [(Var(h), ':ARG1', Var(p)), (Var(p), ':ARG0-of', Var(h))]
        >>> a.role_triples(head=Var('h'))
        [(Var(h), ':ARG1', Var(p))]
        '''
        tt = [(h, r, d) for h, r, d in self.triples(**kwargs)
              if r not in (':instance', ':instance-of', ':top')]
        return tt

    def constants(self):
        return self._constants

    def concept(self, variable):
        return self._v2c[variable]

    def concepts(self):
        return list(self._v2c.items())

    def var2concept(self):
        return dict(self._v2c)

    def alignments(self):
        return dict(self._alignments)

    def tokens(self):
        return self._tokens

    def reentrancies(self):
        '''Counts the number of times each variable is mentioned in the annotation
        beyond the one where it receives a concept. Non-reentrant variables are not
        included in the output.'''
        c = defaultdict(int)
        for h, r, d in self.triples():
            if isinstance(d, Var):
                c[d] += 1
            elif isinstance(d, Concept):
                c[h] -= 1
        # the addition removes non-positive entries
        return Counter(c) + Counter()

    # def __repr__(self):
    # return 'AMR(v2c='+repr(self._v2c)+', triples='+repr(self._triples)+',
    # constants='+repr(self._constants)+')'

    def __call__(self, **kwargs):
        return self.__str__(**kwargs)

    def __str__(
            self,
            alignments=True,
            tokens=True,
            compressed=False,
            indent=' ' * 4):
        return self._anno

    def dfs(self):
        '''
        Assumes triples are stored in a sensible order (reflecting how they are encountered in a valid AMR).

        >>> a = AMR('(p / person :ARG0-of (h / hug-01 :ARG0 p :ARG1 p) :mod (s / strange))')
        >>> # people who hug themselves and are strange
        >>> print(str(a))
        (p / person
            :ARG0-of (h / hug-01
                :ARG0 p
                :ARG1 p)
            :mod (s / strange))
        '''
        variables = dict()
        checkleaf = False
        s = []
        for h, r, d in self.triples():
            if r == ':instance-of':
                continue
            if d.is_var():
                s.append((r,self._v2c[d],h,d))   #relation, concept, original head, dependent
            elif d.is_constant() :
                s.append((r,d,h,d))
        out = []
        for i in range(len(s)):
            out.append((s[i][0],s[i][1]))
        return out




    def __repr__(self):
        return self.__str__()


good_tests = [
    '''(h / hot)''',
    '''(h / hot :mode expressive)''',
    '''(h / hot :mode "expressive")''',
    '''(h / hot :domain h)''',
    '''  (  h  /  hot   :mode  expressive  )   ''',
    '''  (  h
/
hot
:mode
expressive
)   ''',
    '''(h / hot
     :mode expressive
     :mod (e / emoticon
          :value ":)"))''',
    '''(n / name :op1 "Washington")''',
    '''(s / state :name (n / name :op1 "Washington"))''',
    '''(s / state :name (n / name :op1 "Ohio"))
''',
    '''(s / state :name (n / name :op1 "Washington")
    )
''',
    '''(s / state
:name (n / name :op1 "Washington"))''',
    '''(s / state :name (n / name :op1 "Washington")
    :wiki "http://en.wikipedia.org/wiki/Washington_(state)")
''',
    '''(f / film :name (n / name :op1 "Victor/Victoria")
    :wiki "http://en.wikipedia.org/wiki/Victor/Victoria_(1995_film)")
''',
    '''(g / go-01 :polarity -
      :ARG0 (b / boy))''',
    '''(a / and
:op1 (l / love-01 :ARG0 (b / boy) :ARG1 (g / girl))
:op2 (l2 / love-01 :ARG0 g :ARG1 b)
)''',
    '''(d / date-entity :month 2 :day 29 :year 2012 :time "16:30" :timezone "PST"
           :weekday (w / wednesday))''',
    '''(a / and :op1 (d / day :quant 40) :op2 (n / night :quant 40))''',
    '''(e / earthquake
           :location (c / country-region :wiki "Tōhoku_region"
                 :name (n / name :op1 "Tohoku"))
           :quant (s / seismic-quantity :quant 9.3)
           :time (d / date-entity :year 23 :era "Heisei"
                 :calendar (c2 / country :wiki "Japan"
                       :name (n2 / name :op1 "Japan"))))''',
    ''' (d / date-entity :polite +
           :time (a / amr-unknown))'''
]

sembad_tests = [    # not a syntax error, but malformed in terms of variables
    '''(h / hot :mod (h / hot))''',
    '''(h / hot :mod q)'''
]

bad_tests = [
    '''h / hot :mode expressive''',
    '''(hot :mode expressive)''',
    '''(h/hot :mode expressive)''',
    '''(h / hot :mode )''',
    '''(h / hot :mode expressive''',
    '''(h / hot :mode (expressive))''',
    '''(h / hot :mode (e / ))''',
    '''((h / hot :mode expressive)''',
    '''(h / hot :mode expressive)

x''',
    '''(s / state :name (n / name :op1 "  Washington  "))''',
    '''(s / state :name (n / name :op1 "Washington")

    )
''',
    '''(s / state

:name (n / name :op1 "Washington"))''',
    '''(e / earthquake
           :location (c / country-region :wiki "Tōhoku_region"
                 :name (n / name :op1 "Tohoku"))
           :quant (s / seismic-quantity :quant 9.3.1)
           :time (d / date-entity :year 23 :era "Heisei"
                 :calendar (c2 / country :wiki "Japan"
                       :name (n2 / name :op1 "Japan"))))'''
]


def test():
    for good in good_tests:
        try:
            a = AMR(good)
        except AMRSyntaxError:
            print('Should be valid!')
            print(sembad)
        except AMRError:
            print('Should be valid!')
            print(sembad)

    for sembad in sembad_tests:
        try:
            a = AMR(sembad)
        except AMRSyntaxError:
            print('Parse should work!')
            print(sembad)
        except AMRError:
            pass    # should trigger exception
        else:
            print('Should be invalid!')
            print(sembad)

    for bad in bad_tests:
        try:
            a = AMR(bad)
        except AMRSyntaxError:
            pass
        else:
            print('Parse should fail!')
            print(bad)

if __name__ == '__main__':
    test()
    import doctest
    doctest.testmod()
