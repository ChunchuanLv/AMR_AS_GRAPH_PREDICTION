import os,re

WORD_EMBEDDING = "WORD_EMBEDDING"
LEMMA_EMBEDDING = "LEMMA_EMBEDDING"
CONCEPT_EMBEDDING = "CONCEPT_EMBEDDING"
WORD_TO_ID = "WORD_TO_ID"
LEMMA_TO_ID= "LEMMA_TO_ID"
CONCEPT_TO_ID = "CONCEPT_TO_ID"
SHORT_LIST_ID = "SHORT_LIST_ID"

ID_TO_WORD = "ID_TO_WORD"
ID_TO_LEMMA ="ID_TO_LEMMA"
ID_TO_CONCEPT = "ID_TO_CONCEPT"

Rule_Frame = "Rule_Frame"
Rule_Constant = "Rule_Constant"
Rule_String = "Rule_String"
Rule_Concept = "Rule_Concept"
Rule_Num = "Rule_Num"

RULE = 0
HIGH = 1
LOW = 2

RE_FRAME_NUM = re.compile(r'-\d\d$')

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


WORD = 0
LEMMA = 1
POS = 2
NER = 3


threshold = 5


# Change the path according to your system
path_to_stanford = os.path.expanduser('~')+"/Dependency"
stanford_classifier = path_to_stanford+'/stanford-ner-2016-10-31/classifiers/english.muc.7class.distsim.crf.ser.gz'
stanford_ner_path = path_to_stanford+'/stanford-ner-2016-10-31/stanford-ner.jar'
stanford_postagger = path_to_stanford+'/stanford-postagger-full-2016-10-31/stanford-postagger.jar'
stanford_postagger_model = path_to_stanford+'/stanford-postagger-full-2016-10-31/models/english-bidirectional-distsim.tagger'

embed_path = os.path.expanduser('~') + "/Data/sskip.100.vectors"

def folder_to_files_path(folder,ends =".txt"):
    files = os.listdir(folder )
    files_path = []
    for f in files:
        if f.endswith(ends):
            files_path.append(folder+f)
    return files_path

trainFolderPath = os.path.expanduser('~')+"/Data/amr_annotation_r2/data/amrs/split/training/"
trainingFilesPath = folder_to_files_path(trainFolderPath)


devFolderPath = os.path.expanduser('~')+"/Data/amr_annotation_r2/data/amrs/split/dev/"
devFilesPath = folder_to_files_path(devFolderPath)

testFolderPath = os.path.expanduser('~')+"/Data/amr_annotation_r2/data/amrs/split/test/"
testFilesPath = folder_to_files_path(testFolderPath)


frame_folder_path = os.path.expanduser('~')+"/Data/amr_annotation_r2/data/frames/propbank-frames-xml-2016-03-08/"
frame_files_path =  folder_to_files_path(frame_folder_path,".xml")

resource_folder_path  = os.path.expanduser('~')+"/Data/amr_annotation_r2/"

have_org_role = resource_folder_path+"have-org-role-91-roles-v1.06.txt"
have_rel_role = resource_folder_path+"have-rel-role-91-roles-v1.06.txt"
morph_verbalization = resource_folder_path+"morph-verbalization-v1.01.txt"
verbalization =  resource_folder_path+"verbalization-list-v1.06.txt"
