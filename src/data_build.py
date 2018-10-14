#!/usr/bin/env python3.6
# coding=utf-8
'''

Scripts build dictionary and data into numbers

Data path information should also be specified here for
trainFolderPath, devFolderPath and testFolderPath
as we allow option to choose from two version of data.

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

from utility.StringCopyRules import *
from utility.ReCategorization import *
from parser.Dict import *

import argparse


def data_build_parser():
    parser = argparse.ArgumentParser(description='data_build.py')

    ## Data options
    parser.add_argument('-threshold', default=10, type=int,
                        help="""threshold for high frequency concepts""")

    parser.add_argument('-jamr', default=0, type=int,
                        help="""wheather to add .jamr at the end""")
    parser.add_argument('-skip', default=0, type=int,
                        help="""skip dict build if dictionary already built""")
    parser.add_argument('-suffix', default=".txt_pre_processed", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('-folder', default=allFolderPath, type=str,
                        help="""the folder""")
    return parser


parser = data_build_parser()

opt = parser.parse_args()

suffix = opt.suffix + "_jamr" if opt.jamr else opt.suffix
with_jamr = "_with_jamr" if opt.jamr else "_without_jamr"
trainFolderPath = opt.folder + "/training/"
trainingFilesPath = folder_to_files_path(trainFolderPath, suffix)

devFolderPath = opt.folder + "/dev/"
devFilesPath = folder_to_files_path(devFolderPath, suffix)

testFolderPath = opt.folder + "/test/"
testFilesPath = folder_to_files_path(testFolderPath, suffix)


def myamr_to_seq(amr, snt_token, lemma_token, pos, rl, fragment_to_node_converter,
                 high_freq):  # high_freq should be a dict()

    def uni_to_list(uni, can_copy=0):
        #    if can_copy: print (uni)
        le = uni.le
        cat = uni.cat  # use right category anyway
        ner = uni.aux
        data = [0, 0, 0, 0, 0]
        data[AMR_AUX] = ner
        data[AMR_LE_SENSE] = uni.sense
        data[AMR_LE] = le
        data[AMR_CAT] = cat
        data[AMR_CAN_COPY] = 1 if can_copy else 0
        return data

    output_concepts = []
    lemma_str = " ".join(lemma_token)
    fragment_to_node_converter.convert(amr, rl, snt_token, lemma_token, lemma_str)
    concepts, rel, root_id = amr.node_value(keys=["value", "align"], all=True)

    results = rl.get_matched_concepts(snt_token, concepts, lemma_token, pos, jamr=opt.jamr)
    aligned_index = []
    n_amr = len(results)
    n_snt = len(snt_token)
    l = len(lemma_token) if lemma_token[-1] != "." else len(lemma_token) - 1

    # hello, linguistic prior here
    old_unaligned_index = [i for i in range(l) if not (
                pos[i] in ["IN", "POS"] or lemma_token[i] == "would" or lemma_token[i] == "will" and pos[i] == "MD"
                or lemma_token[i] == "have" and pos[i] not in ["VB", "VBG"])
                           or lemma_token[i] in ["although", "while", "of", "if", "in", "per", "like", "by", "for"]]

    for i, n_c_a in enumerate(results):
        uni = n_c_a[1]
        align = [a[0] for a in n_c_a[2]] if len(n_c_a[2]) > 0 else old_unaligned_index
        aligned_index += align

        data = uni_to_list(uni, len(n_c_a[2]) > 0)
        data.append(align)
        output_concepts.append(data)
    if len(aligned_index) == 0:
        output_concepts[0][-1] = [int((len(lemma_token) - 1) / 2)]
        aligned_index = [int((len(lemma_token) - 1) / 2)]
    assert len(aligned_index) > 0, (results, amr._anno, " ".join(lemma_token))
    unaligned_index = [i for i in range(n_snt) if i not in aligned_index]  # or [-1 n_snt] for all
    if len(unaligned_index) == 0: unaligned_index = [-1, n_snt]
    #  assert n_snt <= n_amr or unaligned_index != [],(n_amr,n_snt,concepts,snt_token,amr
    for i in range(n_amr, n_snt):
        output_concepts.append([NULL_WORD, NULL_WORD, NULL_WORD, NULL_WORD, 0, [-1, n_snt]])  # len(amr) >= len(snt)
    printed = False
    for i in range(len(output_concepts)):
        if output_concepts[i][-1] == []:
            if not printed:
                print(output_concepts[i])
                print (list(zip(snt_token, lemma_token, pos)))
                print(concepts, amr)
                printed = True
            output_concepts[i][-1] = [-1, n_snt]

    rel_feature = []
    rel_tgt = []
    for amr_index, role_list in rel:
        amr_concept = uni_to_list(amr_index[
                                      0])  # if align else  uni_to_list(AMRUniversal(UNK_WORD,output_concepts[amr_index[1]][AMR_CAT],NULL_WORD))
        rel_feature.append(amr_concept[:4] + [amr_index[1]])
        #     assert amr_index[1] < len(results), (concepts, rel)
        rel_tgt.append(role_list)  # [role,rel_index]
    return output_concepts, [rel_feature, rel_tgt, root_id], unaligned_index  # [[[lemma1,lemma2],category,relation]]


def filter_non_aligned(input_concepts, rel, unaligned_index):
    rel_feature, rel_tgt, root_id = rel

    filtered_index = {}  # original -> filtered

    output_concepts = []
    for i, data in enumerate(input_concepts):
        if len(data[-1]) == 0:
            output_concepts.append(
                [NULL_WORD, NULL_WORD, NULL_WORD, NULL_WORD, 0, unaligned_index])  # len(amr) >= len(snt)
        elif len(data[-1]) == 1 or data[AMR_CAT] == NULL_WORD:
            output_concepts.append(data)
            filtered_index[i] = len(output_concepts) - 1
        else:
            assert False, (i, data, input_concepts, rel)
    out_rel_feature, out_rel_tgt = [], []
    filtered_rel_index = {}  # original -> filtered  for dependency indexing
    for i, data in enumerate(rel_feature):
        index = data[-1]
        if index in filtered_index:
            new_index = filtered_index[index]
            out_rel_feature.append(data[:-1] + [new_index])
            filtered_rel_index[i] = len(out_rel_feature) - 1

    for i, roles in enumerate(rel_tgt):
        if i in filtered_rel_index:
            new_roles = [[role, filtered_rel_index[j]] for role, j in roles if j in filtered_rel_index]
            out_rel_tgt.append(new_roles)

    if root_id not in filtered_rel_index:
        root_id = 0

    assert len(output_concepts) > 0, (input_concepts, rel, unaligned_index)

    return output_concepts, [out_rel_feature, out_rel_tgt, root_id]


def add_seq_to_dict(dictionary, seq):
    for i in seq:
        dictionary.add(i)


def aligned(align_list):
    return align_list[0] == -1


# id_seq :  [(lemma,cat,lemma_sensed,ner])]
def amr_seq_to_id(lemma_dict, category_dict, lemma_sensed_dict, aux_dict, amr_seq):
    id_seq = []
    for l in amr_seq:
        data = [0] * 5
        data[AMR_CAT] = category_dict[l[AMR_CAT]]
        data[AMR_LE] = lemma_dict[l[AMR_LE]]
        data[AMR_AUX] = aux_dict[l[AMR_AUX]]
        data[AMR_SENSE] = sensed_dict[l[AMR_SENSE]]
        data[AMR_CAN_COPY] = l[AMR_CAN_COPY]
        id_seq.append(data)
    return id_seq


def amr_seq_to_dict(lemma_dict, category_dict, sensed_dict, aux_dict, amr_seq):  # le,cat,le_sense,ner,align
    for i in amr_seq:
        category_dict.add(i[AMR_CAT])
        lemma_dict.add(i[AMR_LE])
        aux_dict.add(i[AMR_NER])
        sensed_dict.add(i[AMR_SENSE])


def rel_seq_to_dict(lemma_dict, category_dict, sensed_dict, rel_dict, rel):  # (amr,index,[[role,amr,index]])
    rel_feature, rel_tgt, root_id = rel
    for i in rel_feature:
        category_dict.add(i[AMR_CAT])
        lemma_dict.add(i[AMR_LE])
    #      sensed_dict.add(i[AMR_SENSE])
    for role_list in rel_tgt:
        for role_index in role_list:
            #  assert (role_index[0]==":top"),rel_tgt
            rel_dict.add(role_index[0])


def rel_seq_to_id(lemma_dict, category_dict, sensed_dict, rel_dict, rel):
    rel_feature, rel_tgt, root_id = rel
    feature_seq = []
    index_seq = []
    roles_mat = []
    for l in rel_feature:
        data = [0] * 3
        data[0] = category_dict[l[AMR_CAT]]
        data[1] = lemma_dict[l[AMR_LE]]
        data[2] = sensed_dict[l[AMR_SENSE]]
        feature_seq.append(data)
        index_seq.append(l[-1])
    for role_list in rel_tgt:
        roles_id = []
        for role_index in role_list:
            roles_id.append([role_index[0], role_index[1]])
        roles_mat.append(roles_id)

    return feature_seq, index_seq, roles_mat, root_id


def handle_sentence(data, filepath, build_dict, n, word_only):
    if n % 1000 == 0:
        print (n)

    ner = data["ner"]
    snt_token = data["tok"]
    pos = data["pos"]
    lemma_token = data["lem"]
    amr_t = data["amr_t"]

    if build_dict:
        if word_only:
            add_seq_to_dict(word_dict, snt_token)
        else:
            add_seq_to_dict(word_dict, snt_token)
            add_seq_to_dict(lemma_dict, lemma_token)
            add_seq_to_dict(pos_dict, pos)
            add_seq_to_dict(ner_dict, ner)
            amr = AMRGraph(amr_t)
            amr_seq, rel, unaligned_index = myamr_to_seq(amr, snt_token, lemma_token, pos, rl,
                                                         fragment_to_node_converter, high_freq)
            amr_seq_to_dict(lemma_dict, category_dict, sensed_dict, aux_dict, amr_seq)
            rel_seq_to_dict(lemma_dict, category_dict, sensed_dict, rel_dict, rel)
    else:
        amr = AMRGraph(amr_t)
        amr_seq, rel, unaligned_index = myamr_to_seq(amr, snt_token, lemma_token, pos, rl, fragment_to_node_converter,
                                                     high_freq)
        if opt.jamr:
            amr_seq, rel = filter_non_aligned(amr_seq, rel, unaligned_index)
        data["snt_id"] = seq_to_id(word_dict, snt_token)[0]
        data["lemma_id"] = seq_to_id(lemma_dict, lemma_token)[0]
        data["pos_id"] = seq_to_id(pos_dict, pos)[0]
        data["ner_id"] = seq_to_id(ner_dict, ner)[0]

        l = len(data["pos_id"])
        if not (l == len(data["snt_id"]) and l == len(data["lemma_id"]) and l == len(data["ner_id"])):
            print (l, len(data["snt_id"]), len(data["lemma_id"]), len(data["ner_id"]))
            print (data["pos_id"])
            print (data["snt_id"])
            print (data["lemma_id"])
            print (data["ner_id"])
            print (pos)
            print (snt_token)
            print (lemma_token)
            print (ner)
            print (data["snt"])
            assert (False)
        data["amr_seq"] = amr_seq
        data["convertedl_seq"] = amr.node_value()
        data["rel_seq"], data["rel_triples"] = amr.get_gold()
        data["amr_id"] = amr_seq_to_id(lemma_dict, category_dict, sensed_dict, aux_dict, amr_seq)
        data["amr_rel_id"], data["amr_rel_index"], data["roles_mat"], data["root"] = rel_seq_to_id(lemma_dict,
                                                                                                   category_dict,
                                                                                                   sensed_dict,
                                                                                                   rel_dict, rel)

        for i in data["amr_rel_index"]:
            assert i < len(data["amr_id"]), (data["amr_rel_index"], amr_seq, data["amr_id"])
        data["index"] = [all[-1] for all in amr_seq]


def readFile(filepath, build_dict=False, word_only=False):
    all_data = load_text_jamr(filepath)

    n = 0
    for data in all_data:
        n = n + 1
        handle_sentence(data, filepath, build_dict, n, word_only)
    if not build_dict:
        outfile = Pickle_Helper(re.sub(end, ".pickle" + with_jamr, filepath))
        outfile.dump(all_data, "data")
        outfile.save()
    return len(all_data)


# Creating ReUsable Object
rl = rules()
rl.load("data/rule_f" + with_jamr)
# initializer = lasagne.init.Uniform()
fragment_to_node_converter = ReCategorizor(from_file=True, path="data/graph_to_node_dict_extended" + with_jamr,
                                           training=False, auto_convert_threshold=opt.threshold)
non_rule_set_f = Pickle_Helper("data/non_rule_set")
non_rule_set = non_rule_set_f.load()["non_rule_set"]
threshold = opt.threshold
high_text_num, high_frequency, low_frequency, low_text_num = unmixe(non_rule_set, threshold)
print (
    "initial converted,threshold,len(non_rule_set),high_text_num,high_frequency,low_frequency,low_text_num,high_freq")
high_freq = {**high_text_num, **high_frequency}

# high_freq =high_frequency

print ("initial converted", threshold, len(non_rule_set), len(high_text_num), len(high_frequency), len(low_frequency),
       len(low_text_num), len(high_freq))


def initial_dict(filename, with_unk=False):
    d = Dict(filename)
    d.addSpecial(NULL_WORD)
    if with_unk:
        d.addSpecial(UNK_WORD)
    #        d.addSpecial(BOS_WORD)
    return d


if not opt.skip:
    word_dict = initial_dict("data/word_dict", with_unk=True)
    pos_dict = initial_dict("data/pos_dict", with_unk=True)

    ner_dict = initial_dict("data/ner_dict", with_unk=True)  # from stanford

    high_dict = initial_dict("data/high_dict", with_unk=True)

    lemma_dict = initial_dict("data/lemma_dict", with_unk=True)

    aux_dict = initial_dict("data/aux_dict", with_unk=True)

    rel_dict = initial_dict("data/rel_dict", with_unk=True)

    category_dict = initial_dict("data/category_dict", with_unk=True)
    sensed_dict = initial_dict("data/sensed_dict", with_unk=True)

    # print ("high freq")
    for uni in high_freq:
        le = uni.le
        lemma_dict.add(le)
        high_dict.add(le)
    #       print (le,high_freq[uni][0])

    for filepath in trainingFilesPath:
        print(("reading " + filepath.split("/")[-1] + "......"))
        n = readFile(filepath, build_dict=True)
        print(("done reading " + filepath.split("/")[-1] + ", " + str(n) + " sentences processed"))

    # only to allow fixed word embedding to be used for those data, alternatively we can build a huge word_embedding for all words from GLOVE...
    for filepath in devFilesPath:
        print(("reading " + filepath.split("/")[-1] + "......"))
        n = readFile(filepath, build_dict=True, word_only=True)
        print(("done reading " + filepath.split("/")[-1] + ", " + str(n) + " sentences processed"))

    for filepath in testFilesPath:
        print(("reading " + filepath.split("/")[-1] + "......"))
        n = readFile(filepath, build_dict=True, word_only=True)
        print(("done reading " + filepath.split("/")[-1] + ", " + str(n) + " sentences processed"))

    print ("len(aux_dict),len(rel_dict),threshold", len(aux_dict), len(rel_dict), threshold)

    rel_dict = rel_dict.pruneByThreshold(threshold)
    aux_dict = aux_dict.pruneByThreshold(threshold)
    category_dict = category_dict.pruneByThreshold(threshold)
    # print (rel_dict)
    word_dict.save()
    lemma_dict.save()
    pos_dict.save()
    aux_dict.save()
    ner_dict.save()
    high_dict.save()
    category_dict.save()
    rel_dict.save()
    sensed_dict.save()
else:

    word_dict = Dict("data/word_dict")
    lemma_dict = Dict("data/lemma_dict")
    aux_dict = Dict("data/aux_dict")
    high_dict = Dict("data/high_dict")
    pos_dict = Dict("data/pos_dict")
    ner_dict = Dict("data/ner_dict")
    rel_dict = Dict("data/rel_dict")
    category_dict = Dict("data/category_dict")
    sensed_dict = Dict("data/sensed_dict")

    word_dict.load()
    lemma_dict.load()
    pos_dict.load()
    ner_dict.load()
    rel_dict.load()
    category_dict.load()
    high_dict.load()
    aux_dict.load()
    sensed_dict.save()

fragment_to_node_converter = ReCategorizor(from_file=True, path="data/graph_to_node_dict_extended" + with_jamr,
                                           training=False, ner_cat_dict=aux_dict)
print("dictionary building done")
print("word_dict \t lemma_dict \tpos_dict \tner_dict \thigh_dict\tsensed_dict \tcategory_dict \taux_dict\trel_dict")
print(
len(word_dict), len(lemma_dict), len(pos_dict), len(ner_dict), len(high_dict), len(sensed_dict), len(category_dict),
len(aux_dict), len(rel_dict))

print(("processing development set"))
for filepath in devFilesPath:
    print(("reading " + filepath.split("/")[-1] + "......"))
    n = readFile(filepath, build_dict=False)
    print(("done reading " + filepath.split("/")[-1] + ", " + str(n) + " sentences processed"))

print(("processing test set"))
for filepath in testFilesPath:
    print(("reading " + filepath.split("/")[-1] + "......"))
    n = readFile(filepath, build_dict=False)

print(("processing training set"))
for filepath in trainingFilesPath:
    print(("reading " + filepath.split("/")[-1] + "......"))
    n = readFile(filepath, build_dict=False)
    print(("done reading " + filepath.split("/")[-1] + ", " + str(n) + " sentences processed"))

print ("initial converted,threshold,len(non_rule_set),high_text_num,high_frequency,low_frequency,low_text_num")
print ("initial converted", threshold, len(non_rule_set), len(high_text_num), len(high_frequency), len(low_frequency),
       len(low_text_num))

