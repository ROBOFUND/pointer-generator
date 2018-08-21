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
import tensorflow as tf
from model import SummarizationModel
from decode import BeamSearchDecoder

import neologdn
from natto import MeCab

class MyBatcher(B.Batcher):
    def __init__(self, batch, vocab, hps, single_pass):
        self._data_path = '//fakepath'
        self._vocab = vocab
        self._hps = hps
        self._single_pass = single_pass

        self.batch = batch
        self.count = 0

    def next_batch(self):
        if self.count == 0:
            self.count += 1
            return self.batch
        else:
            return None

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vocab_size', type=int, default=50000)
    p.add_argument('--single_pass', default=False, action='store_true')
    p.add_argument('--json_path', default="", type=str)
    p.add_argument('--pointer_gen', default=True)
    p.add_argument('vocab_path')
    p.add_argument('checkpoint')
    args = p.parse_args()
    return args

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('vocab_size', 50000, '')
tf.app.flags.DEFINE_boolean('single_pass', False, '')
tf.app.flags.DEFINE_string('json_path', '', '')
tf.app.flags.DEFINE_boolean('pointer_gen', True, '')
tf.app.flags.DEFINE_string('vocab_path', '', '')
tf.app.flags.DEFINE_string('log_root', '', '')
tf.app.flags.DEFINE_integer('beam_size', 1, '')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')

def get_hps():
    _hps = {
        'batch_size': 1, # 10
        'mode': 'decode',
        'lr': 0.15,
        'adagrad_init_acc': 0.1,
        'rand_unif_init_mag': 0.02,
        'trunc_norm_init_std': 1e-4,
        'max_grad_norm': 2.0,
        'hidden_dim': 256,
        'emb_dim': 128,
        'max_enc_steps': 400,
        'max_dec_steps': 100,
        'coverage': False,
        'cov_loss_wt': 1.0,
        'pointer_gen': True,
        'beam_size': 1,
    }
    hps = collections.namedtuple('hps', _hps.keys())(**_hps)
    return hps

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
    tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
    args = FLAGS # get_args()
    vocab = Vocab(args.vocab_path, args.vocab_size) # create a vocabulary
    hps = get_hps()
    b = json_batch(args.json_path, hps, vocab)
    batcher = MyBatcher(b, vocab, hps, args.single_pass)

    decode_model_hps = hps._replace(max_dec_steps=1)
    model = SummarizationModel(decode_model_hps, vocab)
    decoder = BeamSearchDecoder(model, batcher, vocab)
    decoder.decode()
    import pdb; pdb.set_trace()
    pass

if __name__ == '__main__':
    main()
