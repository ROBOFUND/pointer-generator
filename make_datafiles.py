# -*- coding: utf-8
#

import argparse
import collections
import s3utils
from tensorflow.core.example import example_pb2
import struct

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

def count_num(fname):
    count = 0
    with open(fname) as f:
        for line in f:
            count += 1
    return count

def get_art_abs(source, target):
    with open(source) as sf, open(target) as tf:
        for src, tgt in zip(sf, tf):
            src = src[:-1]
            tgt = tgt[:-1]
            yield src, tgt

def write_to_bin(source, target, vocab_path, out_file):
    num_stories = count_num(source)
    vocab_counter = collections.Counter()
    with open(out_file, 'wb') as writer:
        gen = get_art_abs(source, target)
        for idx in range(num_stories):
            if idx % 1000 == 0:
                print("Writing story %i of %i; %.2f percent done" %
                      (idx, num_stories, float(idx)*100.0/float(num_stories)))
            article, abstract = next(gen)
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([bytes(article, 'utf-8')])
            tf_example.features.feature['abstract'].bytes_list.value.extend([bytes(abstract, 'utf-8')])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))
            
            art_tokens = article.split(' ')
            abs_tokens = abstract.split(' ')
            abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
            tokens = art_tokens + abs_tokens
            tokens = [t.strip() for t in tokens] # strip
            tokens = [t for t in tokens if t!=""] # remove empty
            tokens = [t for t in tokens if not t.isdigit()] # remove number
            vocab_counter.update(tokens)
    print("Finished writing file %s\n" % out_file)
    print("Writing vocab file...")
    with open(vocab_path, 'w') as writer:
        for word, count in vocab_counter.most_common():
            writer.write(word + ' ' + str(count) + '\n')
    print("Finished writing vocab file")
            

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('source')
    p.add_argument('target')
    p.add_argument('vocab')
    p.add_argument('out')
    args = p.parse_args()
    return args

def main():
    args = get_args()
    write_to_bin(args.source, args.target, args.vocab, args.out)

if __name__ == '__main__':
    main()
