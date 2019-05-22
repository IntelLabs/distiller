PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<s>'
EOS_TOKEN = '<\s>'

PAD, UNK, BOS, EOS = [0, 1, 2, 3]

VOCAB_FNAME = 'vocab.bpe.32000'

SRC_TRAIN_FNAME = 'train.tok.clean.bpe.32000.en'
TGT_TRAIN_FNAME = 'train.tok.clean.bpe.32000.de'

SRC_VAL_FNAME = 'newstest_dev.tok.clean.bpe.32000.en'
TGT_VAL_FNAME = 'newstest_dev.tok.clean.bpe.32000.de'

SRC_TEST_FNAME = 'newstest2014.tok.bpe.32000.en'
TGT_TEST_FNAME = 'newstest2014.tok.bpe.32000.de'

TGT_TEST_TARGET_FNAME = 'newstest2014.de'

DETOKENIZER = 'mosesdecoder/scripts/tokenizer/detokenizer.perl'
