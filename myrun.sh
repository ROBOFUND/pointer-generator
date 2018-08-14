#!/bin/sh

#--mode=train \
    
python -m pdb run_summarization.py \
       --data_path=s3://enc-report-data/pgn/train_data.bin \
       --vocab_path=s3://enc-report-data/pgn/vocab.txt \
       --log_root=../log-pgn --exp_name=mysum
