#!/bin/bash

python -m embedding_algorithms.code2seq.main \
--trainpath train.h5 \
-g 0 \
--validpath valid.h5 \
--datapath java-small.dict.c2s \
-e 40 \
-b 256 \
--savename output \
--trainnum 691974 \
--validnum 23844
