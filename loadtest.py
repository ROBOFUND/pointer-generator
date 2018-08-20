# -*- coding: utf-8 -*-
#
"""
train data binary(pb2)を読み込む
"""

import argparse
import collections
from data import Vocab
from batcher import Batcher

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vocab-size', type=int, default=50000)
    p.add_argument('--single-pass', default=False, action='store_true')
    p.add_argument('data_path')
    p.add_argument('vocab_path')
    args = p.parse_args()
    return args

def main():
    args = get_args()
    vocab = Vocab(args.vocab_path, args.vocab_size) # create a vocabulary
    _hps = {
        'batch_size': 10,
        'mode': 'decode',
        'max_enc_steps': 400,
        'max_dec_steps': 100,
        'pointer_gen': True,
    }
    hps = collections.namedtuple('hps', _hps.keys())(**_hps)
    batcher = Batcher(args.data_path, vocab, hps, args.single_pass)
    import pdb; pdb.set_trace()
    x = batcher.next_batch()
    import pdb; pdb.set_trace()
    pass

if __name__ == '__main__':
    main()
