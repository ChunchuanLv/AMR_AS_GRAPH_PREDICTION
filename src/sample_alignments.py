#!/usr/bin/env python3.6
# coding=utf-8
'''

Scripts to train the model

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

from parser.Dict import *

from parser.DataIterator import *
import parser
import argparse
from utility.data_helper import Pickle_Helper
from src import *
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
from torch import cuda
from torch.autograd import Variable
import time
from utility.Naive_Scores import *
from collections import defaultdict


def train_parser():
    parser = argparse.ArgumentParser(description='train')
    ## Data options
    parser.add_argument('-suffix', default=".txt_pre_processed", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('-folder', default=allFolderPath, type=str,
                        help="""the folder""")
    parser.add_argument('-jamr', default=0, type=int,
                        help="""wheather to use fixed alignment""")
    parser.add_argument('-save_to', default=save_to,
                        help="""folder to save""")
    parser.add_argument('-train_from', default=None,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model.""")
    parser.add_argument('-get_wiki', type=bool, default=True)
    parser.add_argument('-get_sense', type=bool, default=True)

    ## Model optionsinitialize_lemma
    parser.add_argument('-cat_bias', type=int, default=1,
                        help='Wheather bias category')

    parser.add_argument('-lemma_bias', type=int, default=0,
                        help='Wheather bias lemma')

    parser.add_argument('-independent_posterior', type=int, default=1,
                        help='Wheather normalize over all lemmas')

    parser.add_argument('-train_posterior', type=int, default=1,
                        help='keep training posterior')

    parser.add_argument('-alpha', type=float, default=0.1,
                        help='unk with alpha ')

    parser.add_argument('-initialize_word', type=int, default=1,
                        help='Wheather initialize_lemma')

    parser.add_argument('-out_alignments', type=str, required=True,
                        help='where to store the samples')

    # layers
    parser.add_argument('-rel_enlayers', type=int, default=2,
                        help='Number of layers in the rel LSTM encoder/decoder')
    parser.add_argument('-root_enlayers', type=int, default=1,
                        help='Number of layers in the root LSTM encoder/decoder')
    parser.add_argument('-txt_enlayers', type=int, default=1,
                        help='Number of layers in the concept LSTM encoder/decoder')
    parser.add_argument('-amr_enlayers', type=int, default=1,
                        help='Number of layers in the amr LSTM encoder/decoder')

    parser.add_argument('-txt_rnn_size', type=int, default=512)
    parser.add_argument('-rel_rnn_size', type=int, default=512,
                        help='Number of hidden units in the rel/root LSTM encoder/decoder')
    parser.add_argument('-amr_rnn_size', type=int, default=200)

    parser.add_argument('-rel', type=float, default=1.0,
                        help='Wheather train relation/root identification')

    # dimensions
    parser.add_argument('-word_dim', type=int, default=300,
                        help='Word embedding sizes')
    parser.add_argument('-dim', type=int, default=200,
                        help='lemma/high embedding sizes')
    parser.add_argument('-pos_dim', type=int, default=32,
                        help='Pos embedding sizes')
    parser.add_argument('-ner_dim', type=int, default=16,
                        help='Ner embedding sizes')
    parser.add_argument('-cat_dim', type=int, default=32,
                        help='Pos embedding sizes')
    parser.add_argument('-rel_dim', type=int, default=200,
                        help='mixed amr node and text representation dimension')
    # parser.add_argument('-residual',   action="store_true", , type=bool, default=True
    #                     help="Add residual connections between RNN layers.")
    parser.add_argument('-brnn', type=bool, default=True,
                        help='Use a bidirectional encoder')

    ## Optimization options

    parser.add_argument('-weight_decay', type=float, default=0.00001,
                        help='l2 weight_decay')

    parser.add_argument('-train_all', type=bool, default=False,
                        help='wheather to train all parameters. useful when reloading model for train')

    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('-epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')

    parser.add_argument('-optim', default='adam',
                        help="Optimization method. [sgd|adagrad|adadelta|adam]")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Starting learning rate. If adagrad/adadelta/adam is
                        used, then this is the global learning rate. Recommended
                        settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.1""")
    parser.add_argument('-max_grad_norm', type=float, default=10,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('-dropout', type=float, default=0.2,
                        help='Dropout probability; applied between LSTM stacks and some MLPs.')

    parser.add_argument('-sink', type=int, default=10,
                        help='steps of sinkhorn procedure')

    parser.add_argument('-sink_t', type=float, default=1,
                        help='gumbel-sinkhorn temperature')

    parser.add_argument('-prior_t', type=float, default=5,
                        help='prior tempeture for gumbel-sinkhorn')

    parser.add_argument('-sink_re', type=float, default=10,
                        help='gumbel-sinkhorn finite step regularzor penalizing non double-stochaticity')

    parser.add_argument('-learning_rate_decay', type=float, default=0.98,
                        help="""Decay learning rate by this much if (i) perplexity
                        does not decrease on the validation set or (ii) epoch has
                        gone past the start_decay_at_limit""")

    parser.add_argument('-start_decay_at', default=5,
                        help="Start decay after this epoch")

    parser.add_argument('-emb_independent', type=int, default=1,
                        help="""wheather relation system use independent embedding""")

    # GPU
    parser.add_argument('-gpus', nargs='*', default=[0], type=int,
                        help="Use ith gpu, if -1 then load in cpu")  # training probably cannot work on cpu
    parser.add_argument('-from_gpus', nargs='*', default=[0], type=int,
                        help="model load from which gpu, must be specified when current gpu id is different from saving time")

    parser.add_argument('-log_per_epoch', type=int, default=10,
                        help="Print stats at this interval.")

    parser.add_argument('-renyi_alpha', type=float, default=.5,
                        help="parameter of renyi_alpha relaxation, "
                             "which is alternative to hierachical relaxation in the paper")

    # External random seed
    parser.add_argument('-random_seed', type=int, default=None,
                        help="Set a random seed externally")

    return parser


def sample_from_model(model, AmrDecoder, trainData, validData, dicts, optim, best_f1=0, best_epoch=0,
                      always_print=False):
    model.train()
    AmrDecoder.train()

    # This assumes batch_size allways constant with the exception of the last
    max_batch_size = len(trainData[0][0])

    bs = 99999
    aligned_concept_to_tokens = []
    num_batches = (min(len(trainData), bs))
    for i in range(num_batches):

        model.zero_grad()

        # Get elements of the batch
        order, srcBatch, tgtBatch, alginBatch, rel_batch, rel_index_batch, rel_roles_batch, gold_roots, sourceBatch = \
        trainData[i]

        # Sample from posterior over alignments
        # sinkhorn_aligments are (concept, words)
        probBatch, posteriors_likelihood_score, src_enc = model((srcBatch, tgtBatch, alginBatch), rel=False)
        sinkhorn_alignments = posteriors_likelihood_score[0]

        # Process each element in the batch
        batch_size = len(sourceBatch)
        # sentences do not come the same order as the train but an order field
        # relative to the batch is provided. We need to keep track of position
        # to reconstruct original sentence position
        for n in range(batch_size):

            # Extract desired data
            sinkhorn_alignment = unpack(sinkhorn_alignments)[0][:, n, :].data
            word_tokens = sourceBatch[n][0]
            unrecategorized_nodes = sourceBatch[n][4]
            unrecategorized_edges = sourceBatch[n][5]
            recategorized_nodes = sourceBatch[n][7]
            num_tokens = len(word_tokens)

            # nuber of concepts (rest are padding with '')
            concept_lemma = [x[1] for x in recategorized_nodes]
            if '' in concept_lemma:
                # until first padding
                num_concepts = concept_lemma.index('')
            else:
                # entire length
                num_concepts = len(concept_lemma)

            # add sinkhorn alignments after rule alignments into
            # recategorized_nodes
            num_align = max(num_tokens, num_concepts)
            for concept_index in range(num_align):
                recategorized_nodes[concept_index].append(
                    sinkhorn_alignment[concept_index, :num_tokens].cpu().numpy()
                )

            # store position of this sentence in the original train file
            position = max_batch_size * i + order[n]

            # TODO:
            # An alignment from unrecategorized_nodes to absolute position in tree
            #
            # Example: if we call the variable node_absolute_position
            #
            # - it has same size as unrecategorized_nodes
            #
            # - one example of content for one right branch ending up in two
            #   leafs at depth 3 and one leaf as left branch of the root
            #
            # node_absolute_position = ["0", "0.0". "0.0.0", "0.0.0.0", "0.0.0.1", "0.1"]
            node_absolute_position =  sourceBatch[n][-1]
       #     print (node_absolute_position)
            # TODO:
            # An alignment from unrecategorized_nodes to recategorized_nodes
            #   rel_index_batch
            # Example: if we call the variable unrec2rec
            #
            # - it has same size as unrecategorized_nodes
            # - its maximum value can not be bigger than size of recategorized_nodes - 1
            #
            # - one example of content (for 6 nodes, with three recategorized
            #   into one)
            #
            # unrec2rec = [0, 1, 2, 2, 2, 3]

            unrec2rec = rel_index_batch[n].data.cpu().numpy()
          #  print (unrec2rec)


            amr_text = sourceBatch[n][-2]

            aligned_concept_to_tokens.append((
                position, word_tokens, recategorized_nodes,
                unrecategorized_nodes, unrecategorized_edges,
                node_absolute_position,unrec2rec,
                amr_text
            ))

        if i % 10 == 0:
            print("%d/%d" % (i, num_batches))

    alignments_storage = Pickle_Helper(opt.out_alignments)
    alignments_storage.dump(aligned_concept_to_tokens, 'sinkhorn_sampled_alignments')
    alignments_storage.save()


def embedding_from_dicts(opt, dicts):
    embs = dict()

    def initial_embedding():
        print("loading fixed word embedding")

        word_dict = dicts["word_dict"]
        lemma_dict = dicts["lemma_dict"]
        word_initialized = 0
        lemma_initialized = 0
        word_embedding = nn.Embedding(dicts["word_dict"].size(),
                                      300,  # size of glove dimension
                                      padding_idx=PAD)
        lemma_lut = nn.Embedding(dicts["lemma_dict"].size(),
                                 opt.lemma_dim,
                                 padding_idx=PAD)
        with open(embed_path, 'r') as f:
            for line in f:
                parts = line.rstrip().split()
                id, id2 = word_dict[parts[0]], lemma_dict[parts[0]]
                if id != UNK and id < word_embedding.num_embeddings:
                    tensor = torch.FloatTensor([float(s) for s in parts[-word_embedding.embedding_dim:]]).type_as(
                        word_embedding.weight.data)
                    word_embedding.weight.data[id].copy_(tensor)
                    word_initialized += 1

                if False and id2 != UNK and id2 < lemma_lut.num_embeddings:
                    tensor = torch.FloatTensor([float(s) for s in parts[-lemma_lut.embedding_dim:]]).type_as(
                        lemma_lut.weight.data)
                    lemma_lut.weight.data[id2].copy_(tensor)
                    lemma_initialized += 1

        print ("word_initialized", word_initialized)
        print ("lemma initialized", lemma_initialized)
        print ("word_total", word_embedding.num_embeddings)
        return word_embedding, lemma_lut

    #  print (dicts["sensed_dict"])
    if opt.initialize_word:
        word_fix_lut, lemma_lut = initial_embedding()
        word_fix_lut.requires_grad = False  ##need load
    else:
        word_fix_lut = nn.Embedding(dicts["word_dict"].size(),
                                    300,
                                    padding_idx=PAD)
        lemma_lut = nn.Embedding(dicts["lemma_dict"].size(),
                                 opt.lemma_dim,
                                 padding_idx=PAD)

    high_lut = nn.Embedding(dicts["high_dict"].size(),
                            opt.lemma_dim,
                            padding_idx=PAD)

    pos_lut = nn.Embedding(dicts["pos_dict"].size(),
                           opt.pos_dim,
                           padding_idx=PAD)
    ner_lut = nn.Embedding(dicts["ner_dict"].size(),
                           opt.ner_dim,
                           padding_idx=PAD)

    rel_lut = nn.Embedding(dicts["rel_dict"].size(),
                           1)  # not actually used, but handy to pass number of relations
    cat_lut = nn.Embedding(dicts["category_dict"].size(),
                           opt.cat_dim,
                           padding_idx=PAD)

    aux_lut = nn.Embedding(dicts["aux_dict"].size(),
                           1,
                           padding_idx=PAD)

    if opt.cuda:
        word_fix_lut.cuda()
        lemma_lut.cuda()
        pos_lut.cuda()
        ner_lut.cuda()
        cat_lut.cuda()
        aux_lut.cuda()
        high_lut.cuda()

    embs["word_fix_lut"] = word_fix_lut
    embs["aux_lut"] = aux_lut
    embs["high_lut"] = high_lut
    embs["lemma_lut"] = lemma_lut
    embs["pos_lut"] = pos_lut
    embs["ner_lut"] = ner_lut
    embs["cat_lut"] = cat_lut
    embs["rel_lut"] = rel_lut

    return embs


def main():
    dicts = read_dicts()
    print('Building model...')
    AmrDecoder = parser.AMRProcessors.AMRDecoder(opt, dicts)

    with_jamr = "_with_jamr" if opt.jamr else "_without_jamr"
    suffix = ".pickle" + with_jamr + "_processed"
    trainFolderPath = opt.folder + "/training/"
    trainingFilesPath = folder_to_files_path(trainFolderPath, suffix)
    assert trainingFilesPath, "No %s file found in %s" % (suffix, trainFolderPath)

    devFolderPath = opt.folder + "/dev/"
    devFilesPath = folder_to_files_path(devFolderPath, suffix)
    assert devFilesPath, "No %s file found in %s" % (suffix, devFolderPath)

    testFolderPath = opt.folder + "/test/"
    testFilesPath = folder_to_files_path(testFolderPath, suffix)
    assert testFilesPath, "No %s file found in %s" % (suffix, testFolderPath)

    dev_data = DataIterator(devFilesPath, opt, dicts["rel_dict"], volatile=True)
    training_data = DataIterator(trainingFilesPath, opt, dicts["rel_dict"])

    f1 = 0
    if opt.train_from is None:
        embs = embedding_from_dicts(opt, dicts)

        AmrModel = parser.models.AmrModel(opt, embs)
        print(AmrModel)
        parameters_to_train = []
        for p in AmrModel.parameters():
            if p.requires_grad:
                parameters_to_train.append(p)
    else:
        AmrModel, parameters_to_train, optt = load_old_model(dicts, opt)
        opt.start_epoch = 1

    optim = parser.Optim.Optim(
        parameters_to_train, opt.optim, opt.learning_rate, opt.max_grad_norm,
        lr_decay=opt.learning_rate_decay,
        start_decay_at=opt.start_decay_at,
        weight_decay=opt.weight_decay
    )
    print(' * number of training sentences. %d' %
          len(training_data.src))
    print(' * batch  size.. %d' % opt.batch_size)

    nParams = sum([p.nelement() for p in AmrModel.parameters()])
    print('* number of parameters: %d' % nParams)

    sample_from_model(AmrModel, AmrDecoder, training_data, dev_data, dicts, optim, best_f1=f1)


if __name__ == "__main__":
    global opt
    global opt_str
    opt = train_parser().parse_args()

    with_jamr = "_with_jamr" if opt.jamr else "_without_jamr"

    opt.lemma_dim = opt.dim
    opt.high_dim = opt.dim

    # used for deciding saved model name
    opt_str = "gpus_" + str(opt.gpus[0])

    opt.cuda = len(opt.gpus)
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with -cuda")

    if opt.cuda:
        cuda.set_device(opt.gpus[0])

    # Set random seed
    if opt.random_seed:
        torch.manual_seed(opt.random_seed)
    main()