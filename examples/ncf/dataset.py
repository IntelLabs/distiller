import numpy as np
import scipy
import scipy.sparse
import torch
import torch.utils.data
import subprocess
import time
from tqdm import tqdm
import os
import pickle
import logging

msglogger = logging.getLogger()


def wccount(filename):
    out = subprocess.Popen(['wc', '-l', filename],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT
                           ).communicate()[0]
    return int(out.partition(b' ')[0])


class TimingContext(object):
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        msglogger.info(self.desc + ' ... ')
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        msglogger.info('Done in {0:.4f} seconds'.format(end - self.start))
        return True


class CFTrainDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, train_fname, nb_neg):
        self._load_train_matrix(train_fname)
        self.nb_neg = nb_neg

    def _load_train_matrix(self, train_fname):
        pkl_name = os.path.splitext(train_fname)[0] + '_data.pkl'
        npz_name = os.path.splitext(train_fname)[0] + '_mat.npz'

        if os.path.isfile(pkl_name) and os.path.isfile(npz_name):
            msglogger.info('Found saved dataset data structures')
            with TimingContext('Loading data list pickle'), open(pkl_name, 'rb') as f:
                self.data = pickle.load(f)
            with TimingContext('Loading matrix npz'):
                self.mat = scipy.sparse.dok_matrix(scipy.sparse.load_npz(npz_name))
            self.nb_users = self.mat.shape[0]
            self.nb_items = self.mat.shape[1]
        else:
            def process_line(line):
                tmp = line.split('\t')
                return [int(tmp[0]), int(tmp[1]), float(tmp[2]) > 0]

            with TimingContext('Loading CSV file'), open(train_fname, 'r') as file:
                data = list(map(process_line, tqdm(file, total=wccount(train_fname))))

            with TimingContext('Calculating min/max'):
                self.nb_users = max(data, key=lambda x: x[0])[0] + 1
                self.nb_items = max(data, key=lambda x: x[1])[1] + 1

            with TimingContext('Constructing data list'):
                self.data = list(filter(lambda x: x[2], data))

            with TimingContext('Saving data list pickle'), open(pkl_name, 'wb') as f:
                pickle.dump(self.data, f)

            with TimingContext('Building dok matrix'):
                self.mat = scipy.sparse.dok_matrix(
                        (self.nb_users, self.nb_items), dtype=np.float32)
                for user, item, _ in tqdm(data):
                    self.mat[user, item] = 1.

            with TimingContext('Converting to COO matrix and saving'):
                scipy.sparse.save_npz(npz_name, self.mat.tocoo(copy=True))

    def __len__(self):
        return (self.nb_neg + 1) * len(self.data)

    def __getitem__(self, idx):
        if idx % (self.nb_neg + 1) == 0:
            idx = idx // (self.nb_neg + 1)
            return self.data[idx][0], self.data[idx][1], np.ones(1, dtype=np.float32)  # noqa: E501
        else:
            idx = idx // (self.nb_neg + 1)
            u = self.data[idx][0]
            j = torch.LongTensor(1).random_(0, self.nb_items).item()
            while (u, j) in self.mat:
                j = torch.LongTensor(1).random_(0, self.nb_items).item()
            return u, j, np.zeros(1, dtype=np.float32)


def load_test_ratings(fname):
    pkl_name = os.path.splitext(fname)[0] + '.pkl'
    if os.path.isfile(pkl_name):
        with TimingContext('Found test rating pickle file - loading'), open(pkl_name, 'rb') as f:
            res = pickle.load(f)
    else:
        def process_line(line):
            tmp = map(int, line.split('\t')[0:2])
            return list(tmp)
        with TimingContext('Loading test ratings from csv'), open(fname, 'r') as f:
            ratings = map(process_line, tqdm(f, total=wccount(fname)))
            res = list(ratings)
        with TimingContext('Saving test ratings list pickle'), open(pkl_name, 'wb') as f:
            pickle.dump(res, f)

    return res


def load_test_negs(fname):
    pkl_name = os.path.splitext(fname)[0] + '.pkl'
    if os.path.isfile(pkl_name):
        with TimingContext('Found test negatives pickle file - loading'), open(pkl_name, 'rb') as f:
            res = pickle.load(f)
    else:
        def process_line(line):
            tmp = map(int, line.split('\t'))
            return list(tmp)
        with TimingContext('Loading test negatives from csv'), open(fname, 'r') as f:
            negs = map(process_line, tqdm(f, total=wccount(fname)))
            res = list(negs)
        with TimingContext('Saving test negatives list pickle'), open(pkl_name, 'wb') as f:
            pickle.dump(res, f)

    return res
