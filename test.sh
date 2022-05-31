#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python -u ./translate.py -k 10 WAP_params.pkl \
	./data/dictionary.txt \
	./data/2014.pkl \
	./data/2014_caption.txt \
	./result/test_decode_result.txt \
	./result/test.wer
