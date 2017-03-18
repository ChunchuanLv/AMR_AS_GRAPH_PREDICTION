#!/usr/bin/env python2.7
#coding=utf-8
'''

@author: Nathan Schneider (nschneid@inf.ed.ac.uk)
@since: 2015-05-06
'''
from __future__ import print_function
import sys, re, fileinput, codecs
from collections import Counter, defaultdict

from amr import AMR, AMRSyntaxError, AMRError, Concept, AMRConstant

c = defaultdict(Counter)
for ln in fileinput.input():
    try:
        a = AMR(ln)
        for h,r,d in a.role_triples(normalize_inverses=True, normalize_mod=False):
            if a._v2c[h].is_frame():
                c[str(a._v2c[h])][r] += 1
    except AMRSyntaxError as ex:
        print(ex, file=sys.stderr)
    except AMRError as ex:
        print(ex, file=sys.stderr)
    
for f,roles in sorted(c.items()):
    print(f,'\t'.join(' '.join([r,str(n)]) for r,n in sorted(roles.items())), sep='\t')
