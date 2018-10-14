# AMR AS GRAPH PREDICTION

This repository contains code for training and using the Abstract Meaning Representation model described in:
[AMR Parsing as Graph Prediction with Latent Alignment](https://arxiv.org/pdf/1805.05286.pdf)

If you use our code, please cite our paper as follows:  
  > @inproceedings{Lyu2018AMRPA,  
  > &nbsp; &nbsp; title={AMR Parsing as Graph Prediction with Latent Alignment},  
  > &nbsp; &nbsp; author={Chunchuan Lyu and Ivan Titov},  
  > &nbsp; &nbsp; booktitle={Proceedings of the Annual Meeting of the Association for Computational Linguistics},  
  > &nbsp; &nbsp; year={2018}  
  > }  

## Prerequisites:
* Python 3.6 
* Stanford Corenlp
* pytorch 0.20
* [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings  
* [AMR dataset and resources files](https://amr.isi.edu/download.html)

##Configuration:
* Set up [Stanford Corenlp server](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html), which feature extraction relies on.
* Change file paths in utility/constants.py accordingly.


##Preprocessing:
Combine all *.txt files into a single one, and use stanford corenlp to extract ner, pos and lemma.
Processed file saved in the same folder. 
`python src/preprocessing.py `
or Process from [AMR-to-English aligner](https://www.isi.edu/natural-language/mt/amr_eng_align.pdf) using java script in AMR_FEATURE (I used eclipse to run it)

Build the copying dictionary and recategorization system (can skip as they are in data/).
`python src/rule_system_build.py `
Build data into tensor.
`python src/data_build.py `

##Training:
Default model is saved in [save_to]/gpus_0valid_best.pt . (save_to is defined in constants.py)
`python src/train.py `

##Testing
Load model to parse from pre-build data.
`python src/generate.py -train_from [gpus_0valid_best.pt]`

##Evaluation
Please use [amr-evaluation-tool-enhanced](https://github.com/ChunchuanLv/amr-evaluation-tool-enhanced).
This is based on Marco Damonte's [amr-evaluation-tool](https://github.com/mdtux89/amr-evaluation)
But with correction concerning unlabeled edge score.

##Parsring
Parse a file where each line consists of a single sentence, output saved at `[file]_parsed`
`python src/parse.py -train_from [gpus_0valid_best.pt] -input [file]`
or
Parse a sentence where each line consists of a single sentence, output saved at `[file]_parsed`
`python src/parse.py -train_from [gpus_0valid_best.pt] -text [type sentence here]`

## Pretrained models
Keeping the files under data/ folder unchanged, download [model](https://drive.google.com/open?id=1KkKKDQdRdXgGJ8w_HhbghNK4ceLZcGvS)
Should allow one to run parsing.

##Notes
This "python src/preprocessing.py" starts with sentence original AMR files, while the paper version is trained on tokenized version provided by [AMR-to-English aligner](https://www.isi.edu/natural-language/mt/amr_eng_align.pdf)
So the results could be slightly different. Also, to build a parser for out of domain data, please start preprocessing with "python src/preprocessing.py" to make everything consistent.

## Contact
Contact (chunchuan.lv@gmail.com) if you have any questions!

