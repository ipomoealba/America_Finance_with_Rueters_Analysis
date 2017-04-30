#!/usr/bin/python3
# -*- encoding:utf-8 -*-
import sys
import os
import logging
import numpy as np
import pymongo
import datetime
import nltk

from nltk import *
from nltk.tokenize import RegexpTokenizer
from pymongo import MongoClient
from sklearn import preprocessing
from nltk.corpus import stopwords

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import conf.settings as config

# logging config
logging.basicConfig(level=logging.INFO)


# data start time and end time
start = config.START
end = config.END

# Data time range
delta = end - start

client = MongoClient(config.MONGO_SERVER, config.MONGO_PORT)
reuters_raw = client.reuters_db
deal_data = client.deal_data


def split_text(data):
    """
    Spilt sentence to tokens and remove stopwords and punctuations
    """
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(str(data))
    filtered_words = [token for token in tokens if token.lower(
    ) not in stopwords.words('english')]
    return filtered_words


def unusual_words(text):
    """
    Detect typo
    """
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)


def lookahead(iterable):
    """
    Pass through all values from the given iterable, augmented by the
    information if there are more values to come after the current one
    (True), or if it is the last value (False).
    """
    # Get an iterator and pull the first value.
    it = iter(iterable)
    last = next(it)
    # Run the iterator to exhaustion (starting from the second value).
    for val in it:
        # Report the *previous* value (more to come).
        yield last, True
        last = val
    # Report the last value.
    yield last, False


def insert2db(raws, method, db_collections, start_date):
    data_list = []
    for t, hasmore in lookahead(raws):
        data_list += (method(t['article_text']))
        if not hasmore:
            logging.info("done insert " + str(start_date) + " data to mongo! ")
            db_collections.insert_one({
                "date": start_date,
                "tokens": data_list
            })


def _tokens_inserter():
    for i in range(delta.days + 1):
        start_date = start + datetime.timedelta(days=i)
        end_date = start + datetime.timedelta(days=(i + 1))

        raws = reuters_raw.business.find({
            "publish_date": {
                "$gte": start_date,
                "$lt": end_date
            }
        })
        insert2db(raws, split_text, deal_data.daily_tokens, start_date)


if __name__ == '__main__':
    _tokens_inserter()
