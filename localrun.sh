#!/bin/sh

#--mode=train \
    
python -m pdb run_summarization.py \
       --data_path=$HOME/data/train_data-2nd.bin \
       --vocab_path=$HOME/data/vocab-2nd.txt \
       --log_root=../log-pgn --exp_name=mysum-2nd
