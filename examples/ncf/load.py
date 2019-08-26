from collections import namedtuple

import pandas as pd


RatingData = namedtuple('RatingData',
                        ['items', 'users', 'ratings', 'min_date', 'max_date'])


def describe_ratings(ratings):
    info = RatingData(items=len(ratings['item_id'].unique()),
                      users=len(ratings['user_id'].unique()),
                      ratings=len(ratings),
                      min_date=ratings['timestamp'].min(),
                      max_date=ratings['timestamp'].max())
    print("{ratings} ratings on {items} items from {users} users"
          " from {min_date} to {max_date}"
          .format(**(info._asdict())))
    return info


def process_movielens(ratings, sort=True):
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    if sort:
        ratings.sort_values(by='timestamp', inplace=True)
    describe_ratings(ratings)
    return ratings


def load_ml_100k(filename, sort=True):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(filename, sep='\t', names=names)
    return process_movielens(ratings, sort=sort)


def load_ml_1m(filename, sort=True):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(filename, sep='::', names=names, engine='python')
    return process_movielens(ratings, sort=sort)


def load_ml_10m(filename, sort=True):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(filename, sep='::', names=names, engine='python')
    return process_movielens(ratings, sort=sort)


def load_ml_20m(filename, sort=True):
    ratings = pd.read_csv(filename)
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    names = {'userId': 'user_id', 'movieId': 'item_id'}
    ratings.rename(columns=names, inplace=True)
    return process_movielens(ratings, sort=sort)


DATASETS = [k.replace('load_', '') for k in locals().keys() if "load_" in k]


def get_dataset_name(filename):
    for dataset in DATASETS:
        if dataset in filename.replace('-', '_').lower():
            return dataset
    raise NotImplementedError


def implicit_load(filename, sort=True):
    func = globals()["load_" + get_dataset_name(filename)]
    return func(filename, sort=sort)
