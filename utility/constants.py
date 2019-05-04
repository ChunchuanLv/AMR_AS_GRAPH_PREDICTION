'''Global constants, and file paths'''
import os,re

# Change the path according to your system

save_to = '/disk/scratch/s1544871/model/'    #the folder amr model will be saved to  (model name is parameterized by some hyper parameter)
train_from = '/disk/scratch/s1544871/model/gpus_0valid_best.pt'  #default model loading
embed_path = "/disk/scratch/s1544871/glove.840B.300d.txt"    #file containing glove embedding
core_nlp_url = 'http://localhost:9000'     #local host url of standford corenlp server
root_path = "/disk/scratch/s1544871"
allFolderPath = root_path + "/amr_annotation_r2/data/alignments/split"
resource_folder_path  = root_path +"/amr_annotation_r2/"
frame_folder_path = resource_folder_path+"data/frames/propbank-frames-xml-2016-03-08/"
have_org_role = resource_folder_path+"have-org-role-91-roles-v1.06.txt"   #not used
have_rel_role = resource_folder_path+"have-rel-role-91-roles-v1.06.txt"   #not used
morph_verbalization = resource_folder_path+"morph-verbalization-v1.01.txt"  #not used
verbalization =  resource_folder_path+"verbalization-list-v1.06.txt"


PAD = 0
UNK = 1

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
NULL_WORD = ""
UNK_WIKI = '<WIKI>'
Special = [NULL_WORD,UNK_WORD,PAD_WORD]
#Categories
Rule_Frame = "Frame"
Rule_Constant = "Constant"
Rule_String = "String"
Rule_Concept = "Concept"
Rule_Comp = "COMPO"
Rule_Num = "Num"
Rule_Re = "Re"    #corenference
Rule_Ner = "Ner"
Rule_B_Ner = "B_Ner"
Rule_Other = "Entity"
Other_Cats = {"person","thing",}
COMP = "0"
Rule_All_Constants = [Rule_Num,Rule_Constant,Rule_String,Rule_Ner]
Splish = "$£%%££%£%£%£%"
Rule_Basics = Rule_All_Constants + [Rule_Frame,Rule_Concept,UNK_WORD,BOS_WORD,EOS_WORD,NULL_WORD,PAD_WORD]

RULE = 0
HIGH = 1
LOW = 2

RE_FRAME_NUM = re.compile(r'-\d\d$')
RE_COMP = re.compile(r'_\d$')
end= re.compile(".txt\_[a-z]*")
epsilon = 1e-8

TXT_WORD = 0
TXT_LEMMA = 1
TXT_POS = 2
TXT_NER = 3


AMR_CAT = 0
AMR_LE = 1
AMR_NER = 2
AMR_AUX = 2
AMR_LE_SENSE = 3
AMR_SENSE = 3
AMR_CAN_COPY = 4

threshold = 5


