# -*- coding: utf-8 -*-
#

"""
一記事の要約
"""

import argparse
import collections
import json
from data import Vocab
from batcher import Batcher
import batcher as B

import neologdn
from natto import MeCab

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vocab-size', type=int, default=50000)
    p.add_argument('--single-pass', default=False, action='store_true')
    p.add_argument('--json-path', default="", type=str)
    p.add_argument('vocab_path')
    p.add_argument('checkpoint')
    args = p.parse_args()
    return args

def json_batch(fname, hps, vocab):
    with open(fname) as f:
        art = json.load(f)
    article = neologdn.normalize(art['body'])
    abstract = neologdn.normalize(art['title'])
    m = MeCab('-Owakati')
    parsed_article = m.parse(article)
    abs_words = m.parse(abstract).split()
    ex = B.Example(parsed_article, abs_words, vocab, hps)
    b = B.Batch([ex], hps, vocab)
    return b

def main():
    args = get_args()
    vocab = Vocab(args.vocab_path, args.vocab_size) # create a vocabulary
    _hps = {
        'batch_size': 1, # 10
        'mode': 'decode',
        'max_enc_steps': 400,
        'max_dec_steps': 100,
        'pointer_gen': True,
    }
    hps = collections.namedtuple('hps', _hps.keys())(**_hps)
    b = json_batch(args.json_path, hps, vocab)
    import pdb; pdb.set_trace()
    pass

if __name__ == '__main__':
    main()