# -*- coding: utf-8 -*-
#
"""
train data binary(pb2)を読み込む
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
    p.add_argument('--data_path', default="", type=str)
    p.add_argument('--json-path', default="", type=str)
    p.add_argument('vocab_path')
    args = p.parse_args()
    return args

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
    if not args.data_path == "":
        batcher = Batcher(args.data_path, vocab, hps, args.single_pass)
        import pdb; pdb.set_trace()
        x = batcher.next_batch()
        import pdb; pdb.set_trace()
        pass
    else:
        with open(args.json_path) as f:
            art = json.load(f)
        article = neologdn.normalize(art['body'])
        abstract = neologdn.normalize(art['title'])
        m = MeCab('-Owakati')
        parsed_article = m.parse(article)
        abs_words = m.parse(abstract).split()
        ex = B.Example(parsed_article, abs_words, vocab, hps)
        b = B.Batch([ex], hps, vocab)
        import pdb; pdb.set_trace()
        pass

if __name__ == '__main__':
    main()
