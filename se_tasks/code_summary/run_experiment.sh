#!/bin/bash

RES_DIR='se_tasks/code_summary/result'
if [ ! -d $RES_DIR ]; then
  mkdir $RES_DIR
else
  echo dir exist
fi


EPOCHS=20
BATCH=128
LR=0.005

TK_PATH='dataset/java-small-preprocess/tk.pkl'
TRAIN_DATA='dataset/java-small-preprocess/train.pkl'  #file for training dataset
TEST_DATA='dataset/java-small-preprocess/test.pkl'    #file for testing dataset



# EMBEDDING_TYPE=1
# EMBEDDING_DIM=100                 #dimension of vectors
# EMBEDDING_PATH='/'                #file for pre-trained vectors
# EXPERIMENT_NAME='best_case'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=3 python -m se_tasks.code_summary.scripts.main --tk_path=$TK_PATH --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG



# EMBEDDING_TYPE=2
# EMBEDDING_DIM=100
# EMBEDDING_PATH='/'
# EXPERIMENT_NAME='worst_case'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=3 python -m se_tasks.code_summary.scripts.main --tk_path=$TK_PATH --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG



# EMBEDDING_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec100_1/doc2vec.vec'
# EXPERIMENT_NAME='100_1_doc2vec'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=3 python -m se_tasks.code_summary.scripts.main --tk_path=$TK_PATH --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG


# EMBEDDING_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec100_1/word2vec.vec'
# EXPERIMENT_NAME='100_1_word2vec'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=3 python -m se_tasks.code_summary.scripts.main --tk_path=$TK_PATH --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG



# EMBEDDING_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec100_1/fasttext.vec'
# EXPERIMENT_NAME='100_1_fasttext'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=3 python -m se_tasks.code_summary.scripts.main --tk_path=$TK_PATH --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG



EMBEDDING_TYPE=0
EMBEDDING_DIM=100
EMBEDDING_PATH='embedding_vec100_1/glove.vec'
EXPERIMENT_NAME='100_1_glove'
EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
echo $EXPERIMENT_NAME
CUDA_VISIBLE_DEVICES=1 python -m se_tasks.code_summary.scripts.main --tk_path=$TK_PATH --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
--embed_dim=$EMBEDDING_DIM --embed_path=$EMBEDDING_PATH \
--train_data=$TRAIN_DATA --test_data=$TEST_DATA --embed_type=$EMBEDDING_TYPE \
--experiment_name=$EXPERIMENT_NAME #| tee $EXPERIMENT_LOG