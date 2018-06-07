__author__ = 's1544871'
from utility.constants import *
from utility.amr import AMRUniversal

class ScoreHelper:

    def __init__(self,name, filter ,second_filter=None):
        self.t_p_tp = [0,0,0]
        self.name = name
        self.f = filter
        self.second_filter = second_filter
        self.false_positive = {}
        self.false_negative = {}

    def T_P_TP_Batch(self,hypos,golds,accumulate=True,second_filter_material =None):
        if self.second_filter:
            T,P,TP,fp,fn = T_P_TP_Batch(hypos,golds,self.f,self.second_filter,second_filter_material)
        else:
    #        assert self.name != "Unlabled SRL Triple",(hypos[-20],"STOP!",golds[-20])
            T,P,TP,fp,fn = T_P_TP_Batch(hypos,golds,self.f)
        if accumulate:
            self.add_t_p_tp(T,P,TP)
            self.add_content(fp,fn)
        return T,P,TP

    def add_t_p_tp(self,T,P,TP):
        self.t_p_tp[0] += T
        self.t_p_tp[1] += P
        self.t_p_tp[2] += TP

    def add_content(self,fp,fn ):
        for i in fp:
            self.false_positive[i] = self.false_positive.setdefault(i,0)+1
        for i in fn:
            self.false_negative[i] = self.false_negative.setdefault(i,0)+1

    def show_error(self,t = 5):
        print ("false_positive",[(k,self.false_positive[k]) for  k in sorted(self.false_positive,key=self.false_positive.get) if self.false_positive[k]> t])
        print ("")
        print ("false_negative",[(k,self.false_negative[k]) for  k in sorted(self.false_negative,key=self.false_negative.get) if self.false_negative[k]>t])
    def __str__(self):
        s = self.name+"\nT,P,TP: "+ " ".join([str(i) for i in  self.t_p_tp])+"\nPrecesion,Recall,F1: "+ " ".join([str(i)for i in  P_R_F1(*self.t_p_tp)])
        return s



def filter_mutual(hypo,gold,mutual_filter):
    filtered_hypo = [item for sublist in filter_seq(mutual_filter,hypo) for item in sublist]
    out_hypo = []
    filtered_gold = [item for sublist in filter_seq(mutual_filter,gold) for item in sublist]
    out_gold = []

    for data in hypo:
        d1,d2 = mutual_filter(data)
        if d1 in filtered_gold and d2 in filtered_gold:
            out_hypo.append(data)


    for data in gold:
        d1,d2 = mutual_filter(data)
        if d1 in filtered_hypo and d2 in filtered_hypo:
            out_gold.append(data)

    return out_hypo,out_gold

def list_to_mulset(l):
    s = dict()
    for i in l:
        if isinstance(i,AMRUniversal) and i.le == "i"and i.cat == Rule_Concept :
            s[i] = 1
        else:
            s[i] = s.setdefault(i,0)+1
    return s

def legal_concept(uni):
    if isinstance(uni,AMRUniversal):
        return (uni.cat,uni.le,uni.sense) if not uni.le in Special and  not uni.cat in Special else None
    else:
        return uni

def nonsense_concept(uni):
    return (uni.cat,uni.le) if not uni.le in Special and  not uni.cat in Special else None

def dynamics_filter(triple,concept_seq):
    if triple[0] in concept_seq and triple[1] in concept_seq or BOS_WORD in triple[0]:
        return triple[:3]

 #   print (triple,concept_seq[0])
    return None

def filter_seq(filter,seq):
    out = []
    for t in seq:
        filtered = filter(t)
        if filtered  and  filtered[0] != BOS_WORD and filtered != BOS_WORD:
            out.append(filtered)
    return out

def remove_sense(uni):
    return (uni.cat,uni.le)

def T_TP_Seq(hypo,gold,filter,second_filter = None,second_filter_material = None):
    gold = filter_seq(filter,gold)
    hypo = filter_seq(filter,hypo)
    fp = []
    fn = []
    if second_filter:  #only for triple given concept


        second_filter_predicated = filter_seq(legal_concept,second_filter_material[0])
        second_filter_with_material = lambda x: second_filter(x,second_filter_predicated)
        gold = filter_seq(second_filter_with_material,gold)


        second_filter_gold = filter_seq(legal_concept,second_filter_material[1])
        second_filter_with_material = lambda x: second_filter(x,second_filter_gold)

        hypo = filter_seq(second_filter_with_material,hypo)

    if len(gold)>0 and isinstance(gold[0],tuple) and len(gold[0])==3 and False:
        print ("")
        print ("source based prediction")
        for t in hypo:
            print (t)
        print ("")
        print ("source gold seq")
        for t in gold:
            print (t)
        print ("")
    TP = 0
    T = len(gold)
    P = len(hypo)
    gold = list_to_mulset(gold)
    hypo = list_to_mulset(hypo)
    for d_g in gold:
        if d_g in hypo :
            TP += min(gold[d_g],hypo[d_g])
            fn = fn + [d_g] *min(gold[d_g]-hypo[d_g],0)
        else:
            fn = fn + [d_g] *gold[d_g]
    for d_g in hypo:
        if d_g in gold :
            fp = fp + [d_g] *min(hypo[d_g]-gold[d_g],0)
        else:
            fp = fp + [d_g] *hypo[d_g]
    return T,P,TP,fp,fn

def T_P_TP_Batch(hypos,golds,filter=legal_concept,second_filter=None,second_filter_material_batch = None):
    TP,T,P = 0,0,0
    FP,FN = [],[]
    assert hypos, golds
    for i in range(len(hypos)):
        if second_filter:
            t,p,tp,fp,fn = T_TP_Seq(hypos[i],golds[i],filter,second_filter,(second_filter_material_batch[0][i],second_filter_material_batch[1][i]))
        else:
            t,p,tp,fp,fn = T_TP_Seq(hypos[i],golds[i],filter)
        T += t
        P +=p
        TP += tp
        FP += fp
        FN += fn
    return T,P,TP,FP,FN


def P_R_F1(T,P,TP):
    if TP == 0:
        return 0,0,0
    P = TP/P
    R = TP/T
    F1 = 2.0/(1.0/P+1.0/R)
    return P,R,F1



#naive set overlapping for different kinds of relations
def rel_scores_initial():


    root_filter = lambda t:(legal_concept(t[0]),legal_concept(t[1]),t[2]) if legal_concept(t[0])  and legal_concept(t[1])  and nonsense_concept(t[0]) == (BOS_WORD,BOS_WORD) else None

    root_score =  ScoreHelper("Root",filter=root_filter)

    rel_filter = lambda t:(legal_concept(t[0]),legal_concept(t[1]),t[2]) if legal_concept(t[0])  and legal_concept(t[1])  else None
    rel_score =  ScoreHelper("REL Triple",filter=rel_filter)

    non_sense_rel_filter = lambda t:(nonsense_concept(t[0]),nonsense_concept(t[1]),t[2]) if legal_concept(t[0])  and legal_concept(t[1])  else None
    nonsense_rel_score =  ScoreHelper("Nonsense REL Triple",filter=non_sense_rel_filter)

    unlabeled_filter =lambda t:(legal_concept(t[0]),legal_concept(t[1])) if legal_concept(t[0])  and legal_concept(t[1]) else None

    unlabeled_rel_score =  ScoreHelper("Unlabeled Rel Triple",filter=unlabeled_filter)

    labeled_rel_score_given_concept =  ScoreHelper("REL Triple given concept",filter = rel_filter, second_filter=dynamics_filter)


    un_srl_filter =lambda t:(legal_concept(t[0]),legal_concept(t[1])) if legal_concept(t[0])  and legal_concept(t[1]) and t[2].startswith(':ARG') else None

    un_frame_score =  ScoreHelper("Unlabled SRL Triple",filter=un_srl_filter)

    srl_filter = lambda t:(legal_concept(t[0]),legal_concept(t[1]),t[2]) if legal_concept(t[0])  and legal_concept(t[1]) and t[2].startswith(':ARG') else None
    frame_score =  ScoreHelper("SRL Triple",filter=srl_filter)

    labeled_srl_score_given_concept =  ScoreHelper("SRL Triple given concept",filter = srl_filter, second_filter=dynamics_filter)

    unlabeled_srl_score_given_concept =  ScoreHelper("Unlabeled SRL Triple given concept",filter = un_srl_filter, second_filter=dynamics_filter)

    return [nonsense_rel_score,rel_score,root_score,unlabeled_rel_score,labeled_rel_score_given_concept,frame_score,un_frame_score,labeled_srl_score_given_concept,unlabeled_srl_score_given_concept]


#naive set overlapping for different kinds of concepts
def concept_score_initial(dicts):

    Non_Sense =  ScoreHelper("Non_Sense",filter=nonsense_concept)
    concept_score = ScoreHelper("Full Concept",filter=legal_concept)
    category_score =  ScoreHelper("Category Only",filter=lambda uni:(uni.cat)
        if legal_concept(uni) else None)
    lemma_score =  ScoreHelper("Lemma Only",filter=lambda uni: (uni.le)
        if legal_concept(uni) else None)
    frame_score =  ScoreHelper("Frame Only",filter=lambda uni: (uni.le)
        if legal_concept(uni) and uni.cat==Rule_Frame else None)
    frame_sense_score =  ScoreHelper("Frame Sensed Only",filter=lambda uni: (uni.le,uni.sense)
        if legal_concept(uni) and uni.cat==Rule_Frame else None)
    frame_non_91_score =  ScoreHelper("Frame non 91 Only",filter=lambda uni: (uni.le,uni.sense)
        if legal_concept(uni) and uni.cat==Rule_Frame and "91" not in uni.sense else None)
    high_score =  ScoreHelper("High Freq Only",filter=lambda uni: (uni.le,uni.cat)
        if  uni.le in dicts["high_dict"] and legal_concept(uni)  else None)
    default_score =  ScoreHelper("Copy Only",filter=lambda uni: (uni.le,uni.cat)
        if  uni.le not in dicts["high_dict"] and legal_concept(uni) else None)
    return  [Non_Sense,concept_score,category_score,frame_score,frame_sense_score,frame_non_91_score,lemma_score,high_score,default_score]
