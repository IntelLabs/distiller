# This code is originally from:
#   https://github.com/pytorch/examples/blob/master/word_language_model/main.py
# It contains a fix as per:
#   https://github.com/pytorch/examples/issues/214
# It contains code to support compression (distiller)
#
# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from collections import OrderedDict
import data
import model

# Distiller imports
import os
import sys
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import distiller
import apputils
from distiller.data_loggers import TensorBoardLogger, PythonLogger


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='checkpoint.pth.tar',
                    help='path to save the final model')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

# Distiller-related arguments
SUMMARY_CHOICES = ['sparsity', 'model', 'modules', 'png', 'percentile']
parser.add_argument('--summary', type=str, choices=SUMMARY_CHOICES,
                    help='print a summary of the model, and exit - options: ' +
                    ' | '.join(SUMMARY_CHOICES))
parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                    help='configuration file for pruning the model (default is to use hard-coded schedule)')
parser.add_argument('--momentum', default=0., type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0., type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")


def draw_lang_model_to_file(model, png_fname, dataset):
    """Draw a language model graph to a PNG file.

    Caveat: the PNG that is produced has some problems, which we suspect are due to
    PyTorch issues related to RNN ONNX export.
    """
    try:
        if dataset == 'wikitext2':
            batch_size = 20
            seq_len = 35
            dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to('cuda')
            hidden = model.init_hidden(batch_size)
            dummy_input = (dummy_input, hidden)
        else:
            msglogger.info("Unsupported dataset (%s) - aborting draw operation" % dataset)
            return
        g = apputils.SummaryGraph(model, dummy_input)
        apputils.draw_model_to_file(g, png_fname)
        msglogger.info("Network PNG image generation completed")

    except FileNotFoundError as e:
        msglogger.info("An error has occured while generating the network PNG image.")
        msglogger.info("Please check that you have graphviz installed.")
        msglogger.info("\t$ sudo apt-get install graphviz")
        raise e

###############################################################################
# Load data
###############################################################################

print("Preparing data (this may take several seconds)...")
corpus = data.Corpus(args.data)
print("Done")
# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)

if args.resume:
    with open(args.resume, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        model.rnn.flatten_parameters()
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)


criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        # The line below was fixed as per: https://github.com/pytorch/examples/issues/214
        for i in range(0, data_source.size(0), args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


def train(epoch, optimizer, compression_scheduler=None):
    # Turn on training mode which enables dropout.
    model.train()

    total_samples = train_data.size(0)
    steps_per_epoch = math.ceil(total_samples / args.bptt)

    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    # The line below was fixed as per: https://github.com/pytorch/examples/issues/214
    for batch, i in enumerate(range(0, train_data.size(0), args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)

        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, minibatch_id=batch, minibatches_per_epoch=steps_per_epoch)
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)

        if compression_scheduler:
            # Before running the backward phase, we allow the scheduler to modify the loss
            # (e.g. add regularization loss)
            loss = compression_scheduler.before_backward_pass(epoch, minibatch_id=batch,
                                                              minibatches_per_epoch=steps_per_epoch, loss=loss,
                                                              return_loss_components=False)

        optimizer.zero_grad()
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()

        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, minibatch_id=batch, minibatches_per_epoch=steps_per_epoch)

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            msglogger.info(
                    '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} '
                    '| loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            stats = ('Peformance/Training/',
                OrderedDict([
                    ('Loss', cur_loss),
                    ('Perplexity', math.exp(cur_loss)),
                    ('LR', lr),
                    ('Batch Time', elapsed * 1000)])
                )
            steps_completed = batch + 1
            distiller.log_training_progress(stats, model.named_parameters(), epoch, steps_completed,
                                            steps_per_epoch, args.log_interval, [tflogger])


def export_onnx(path, batch_size, seq_len):
    msglogger.info('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Distiller loggers
msglogger = apputils.config_pylogger('logging.conf', None)
tflogger = TensorBoardLogger(msglogger.logdir)
tflogger.log_gradients = True
pylogger = PythonLogger(msglogger)

if args.summary:
    which_summary = args.summary
    if which_summary == 'png':
        draw_lang_model_to_file(model, 'rnn.png', 'wikitext2')
    elif which_summary == 'percentile':
        percentile = 0.9
        for name, param in model.state_dict().items():
            if param.dim() < 2:
                # Skip biases
                continue
            bottomk, _ = torch.topk(param.abs().view(-1), int(percentile * param.numel()),
                                    largest=False, sorted=True)
            threshold = bottomk.data[-1]
            msglogger.info("parameter %s: q = %.2f" %(name, threshold))
    else:
        distiller.model_summary(model, which_summary, 'wikitext2')
    exit(0)

compression_scheduler = None

if args.compress:
    # Create a CompressionScheduler and configure it from a YAML schedule file
    source = args.compress
    compression_scheduler = distiller.config.file_config(model, None, args.compress)

optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          patience=0, verbose=True, factor=0.5)

# Loop over epochs.
# At any point you can hit Ctrl + C to break out of training early.
best_val_loss = float("inf")
try:
    for epoch in range(0, args.epochs):
        epoch_start_time = time.time()
        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch)

        train(epoch, optimizer, compression_scheduler)

        val_loss = evaluate(val_data)
        msglogger.info('-' * 89)
        msglogger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.3f} | '
                       'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        msglogger.info('-' * 89)

        distiller.log_weights_sparsity(model, epoch, loggers=[tflogger, pylogger])

        stats = ('Peformance/Validation/',
            OrderedDict([
                ('Loss', val_loss),
                ('Perplexity', math.exp(val_loss))]))
        tflogger.log_training_progress(stats, epoch, 0, total=1, freq=1)

        with open(args.save, 'wb') as f:
            torch.save(model, f)

        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val_loss:
            with open(args.save+".best", 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        lr_scheduler.step(val_loss)

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch)

except KeyboardInterrupt:
    msglogger.info('-' * 89)
    msglogger.info('Exiting from training early')

# Load the last saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
msglogger.info('=' * 89)
msglogger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
msglogger.info('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
