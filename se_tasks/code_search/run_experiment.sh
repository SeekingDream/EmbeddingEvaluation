#!/bin/bash

RES_DIR='se_tasks/code_search/result'
if [ ! -d $RES_DIR ]; then
  mkdir $RES_DIR
else
  echo dir exist
fi


BATCH=512
LR=0.005
# TRAIN_DATA='se_tasks/code_search/dataset/train.tsv'
# TEST_DATA='se_tasks/code_search/dataset/test.tsv'


# EMBED_TYPE=1
# EMBEDDING_DIM=100                 #dimension of vectors
# EMBEDDING_PATH='/'                #file for pre-trained vectors
# EXPERIMENT_NAME='best_case'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=0 python -m se_tasks.code_search.scripts.train \
# --embed_type=$EMBED_TYPE --learning_rate=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG



# EMBED_TYPE=2
# EMBEDDING_DIM=100
# EMBEDDING_PATH='/'
# EXPERIMENT_NAME='worst_case'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=1 python -m se_tasks.code_search.scripts.train \
# --embed_type=$EMBED_TYPE --learning_rate=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG



# EMBED_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec100_1/doc2vec.vec'
# EXPERIMENT_NAME='100_1_doc2vec'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=3 python -m se_tasks.code_search.scripts.train \
# --embed_type=$EMBED_TYPE --learning_rate=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG


# EMBED_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec100_1/word2vec.vec'
# EXPERIMENT_NAME='100_1_word2vec'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=2 python -m se_tasks.code_search.scripts.train \
# --embed_type=$EMBED_TYPE --learning_rate=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG



# EMBED_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec100_1/fasttext.vec'
# EXPERIMENT_NAME='100_1_fasttext'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=4 python -m se_tasks.code_search.scripts.train \
# --embed_type=$EMBED_TYPE --learning_rate=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG



# EMBED_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec100_1/glove.vec'
# EXPERIMENT_NAME='100_1_glove'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=2 python -m se_tasks.code_search.scripts.train \
# --embed_type=$EMBED_TYPE --learning_rate=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG



# EMBED_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec100_1/ori_code2seq.vec'
# EXPERIMENT_NAME='code2seq'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=4 python -m se_tasks.code_search.scripts.train \
# --embed_type=$EMBED_TYPE --learning_rate=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG


EMBED_TYPE=0
EMBEDDING_DIM=100
EMBEDDING_PATH='embedding_vec100_1/code2vec.vec'
EXPERIMENT_NAME='code2vec'
EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
echo $EXPERIMENT_NAME
CUDA_VISIBLE_DEVICES=5 python -m se_tasks.code_search.scripts.train \
--embed_type=$EMBED_TYPE --learning_rate=$LR \
--embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
--experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG