import logging
import math
import time
import os
from itertools import cycle

import torch
import torch.optim
import torch.utils.data

from mlperf_compliance import mlperf_log

from seq2seq.train.distributed import DistributedDataParallel as DDP
from seq2seq.train.fp_optimizers import Fp16Optimizer, Fp32Optimizer
from seq2seq.utils import AverageMeter
from seq2seq.utils import sync_workers


class Seq2SeqTrainer:

    def __init__(self, model, criterion, opt_config,
                 print_freq=10,
                 save_freq=1000,
                 grad_clip=float('inf'),
                 batch_first=False,
                 save_info={},
                 save_path='.',
                 checkpoint_filename='checkpoint%s.pth',
                 keep_checkpoints=5,
                 math='fp32',
                 cuda=True,
                 distributed=False,
                 verbose=False):
        super(Seq2SeqTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.epoch = 0
        self.save_info = save_info
        self.save_path = save_path
        self.save_freq = save_freq
        self.save_counter = 0
        self.checkpoint_filename = checkpoint_filename
        self.checkpoint_counter = cycle(range(keep_checkpoints))
        self.opt_config = opt_config
        self.cuda = cuda
        self.distributed = distributed
        self.print_freq = print_freq
        self.batch_first = batch_first
        self.verbose = verbose
        self.loss = None

        if cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        if distributed:
            self.model = DDP(self.model)

        if math == 'fp16':
            self.model = self.model.half()
            self.fp_optimizer = Fp16Optimizer(self.model, grad_clip)
            params = self.fp_optimizer.fp32_params
        elif math == 'fp32':
            self.fp_optimizer = Fp32Optimizer(self.model, grad_clip)
            params = self.model.parameters()

        opt_name = opt_config['optimizer']
        lr = opt_config['lr']
        self.optimizer = torch.optim.__dict__[opt_name](params, lr=lr)
        mlperf_log.gnmt_print(key=mlperf_log.OPT_NAME,
                              value=opt_name)
        mlperf_log.gnmt_print(key=mlperf_log.OPT_LR,
                              value=lr)
        mlperf_log.gnmt_print(key=mlperf_log.OPT_HP_ADAM_BETA1,
                              value=self.optimizer.defaults['betas'][0])
        mlperf_log.gnmt_print(key=mlperf_log.OPT_HP_ADAM_BETA2,
                              value=self.optimizer.defaults['betas'][1])
        mlperf_log.gnmt_print(key=mlperf_log.OPT_HP_ADAM_EPSILON,
                              value=self.optimizer.defaults['eps'])

    def iterate(self, src, tgt, update=True, training=True):
        src, src_length = src
        tgt, tgt_length = tgt
        src_length = torch.LongTensor(src_length)
        tgt_length = torch.LongTensor(tgt_length)

        num_toks = {}
        num_toks['tgt'] = int(sum(tgt_length - 1))
        num_toks['src'] = int(sum(src_length))

        if self.cuda:
            src = src.cuda()
            src_length = src_length.cuda()
            tgt = tgt.cuda()

        if self.batch_first:
            output = self.model(src, src_length, tgt[:, :-1])
            tgt_labels = tgt[:, 1:]
            T, B = output.size(1), output.size(0)
        else:
            output = self.model(src, src_length, tgt[:-1])
            tgt_labels = tgt[1:]
            T, B = output.size(0), output.size(1)

        loss = self.criterion(output.view(T * B, -1).float(),
                              tgt_labels.contiguous().view(-1))

        loss_per_batch = loss.item()
        loss /= B

        if training:
            self.fp_optimizer.step(loss, self.optimizer, update)

        loss_per_token = loss_per_batch / num_toks['tgt']
        loss_per_sentence = loss_per_batch / B

        return loss_per_token, loss_per_sentence, num_toks

    def feed_data(self, data_loader, training=True):
        if training:
            assert self.optimizer is not None
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_per_token = AverageMeter()
        losses_per_sentence = AverageMeter()

        tot_tok_time = AverageMeter()
        src_tok_time = AverageMeter()
        tgt_tok_time = AverageMeter()

        batch_size = data_loader.batch_size

        end = time.time()
        for i, (src, tgt, _) in enumerate(data_loader):
            self.save_counter += 1
            # measure data loading time
            data_time.update(time.time() - end)

            # do a train/evaluate iteration
            stats = self.iterate(src, tgt, training=training)
            loss_per_token, loss_per_sentence, num_toks = stats

            # measure accuracy and record loss
            losses_per_token.update(loss_per_token, num_toks['tgt'])
            losses_per_sentence.update(loss_per_sentence, batch_size)

            # measure elapsed time
            elapsed = time.time() - end
            batch_time.update(elapsed)
            src_tok_time.update(num_toks['src'] / elapsed)
            tgt_tok_time.update(num_toks['tgt'] / elapsed)
            tot_num_toks = num_toks['tgt'] + num_toks['src']
            tot_tok_time.update(tot_num_toks / elapsed)
            self.loss = losses_per_token.avg
            end = time.time()

            if i % self.print_freq == 0:
                phase = 'TRAIN' if training else 'EVAL'
                log = []
                log += [f'{phase} [{self.epoch}][{i}/{len(data_loader)}]']
                log += [f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})']
                log += [f'Data {data_time.val:.3f} ({data_time.avg:.3f})']
                log += [f'Tok/s {tot_tok_time.val:.0f} ({tot_tok_time.avg:.0f})']
                if self.verbose:
                    log += [f'Src tok/s {src_tok_time.val:.0f} ({src_tok_time.avg:.0f})']
                    log += [f'Tgt tok/s {tgt_tok_time.val:.0f} ({tgt_tok_time.avg:.0f})']
                    log += [f'Loss/sentence {losses_per_sentence.val:.1f} ({losses_per_sentence.avg:.1f})']
                log += [f'Loss/tok {losses_per_token.val:.8f} ({losses_per_token.avg:.8f})']
                log = '\t'.join(log)
                logging.info(log)

            save_chkpt = (self.save_counter % self.save_freq) == (self.save_freq - 1)
            if training and save_chkpt:
                self.save_counter = 0
                self.save_info['iteration'] = i
                identifier = next(self.checkpoint_counter, -1)
                if identifier != -1:
                    with sync_workers() as rank:
                        if rank == 0:
                            self.save(identifier=identifier)

        return losses_per_token.avg

    def preallocate(self, data_loader, training):
        batch_size = data_loader.batch_size
        max_len = data_loader.dataset.max_len

        src_length = [max_len] * batch_size
        tgt_length = [max_len] * batch_size

        if self.batch_first:
            shape = (batch_size, max_len)
        else:
            shape = (max_len, batch_size)

        src = torch.full(shape, 4, dtype=torch.int64)
        tgt = torch.full(shape, 4, dtype=torch.int64)
        src = src, src_length
        tgt = tgt, tgt_length
        self.iterate(src, tgt, update=False, training=training)

    def optimize(self, data_loader):
        torch.set_grad_enabled(True)
        self.model.train()
        torch.cuda.empty_cache()
        self.preallocate(data_loader, training=True)
        output = self.feed_data(data_loader, training=True)
        torch.cuda.empty_cache()
        return output

    def evaluate(self, data_loader):
        torch.set_grad_enabled(False)
        self.model.eval()
        torch.cuda.empty_cache()
        self.preallocate(data_loader, training=False)
        output = self.feed_data(data_loader, training=False)
        torch.cuda.empty_cache()
        return output

    def load(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location={'cuda:0': 'cpu'})
            self.model.load_state_dict(checkpoint['state_dict'])
            self.fp_optimizer.initialize_model(self.model)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch = checkpoint['epoch']
            self.loss = checkpoint['loss']
            logging.info(f'loaded checkpoint {filename} (epoch {self.epoch})')
        else:
            logging.error(f'invalid checkpoint: {filename}')

    def save(self, identifier=None, is_best=False, save_all=False):

        def write_checkpoint(state, filename):
            filename = os.path.join(self.save_path, filename)
            logging.info(f'saving model to {filename}')
            torch.save(state, filename)

        state = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': getattr(self, 'loss', None),
        }
        state = dict(list(state.items()) + list(self.save_info.items()))

        if identifier is not None:
            filename = self.checkpoint_filename % identifier
            write_checkpoint(state, filename)

        if is_best:
            filename = 'model_best.pth'
            write_checkpoint(state, filename)

        if save_all:
            filename = f'checkpoint_epoch_{self.epoch:03d}.pth'
            write_checkpoint(state, filename)
