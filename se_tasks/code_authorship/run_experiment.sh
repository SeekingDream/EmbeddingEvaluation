# !/bin/bash
EPOCHS=20
BATCH=32
LR=0.005
FOLDER='se_tasks/code_authorship'
RES_DIR=$FOLDER'/result/'

# make result dir
if [ ! -d $RES_DIR ]; then
  mkdir $RES_DIR
else
  echo dir exist
fi

CLASSES=250                       # number of the category
TRAIN_DATA=$FOLDER'/dataset/train.tsv'  # file for training dataset
TEST_DATA=$FOLDER'/dataset/test.tsv'    # file for testing dataset

# 0 is loading a pre-trained embedding, 
# 1 is training from scratch (best case),
# 2 is use random vectors but not update vectors (worst cases).
# EMBEDDING_TYPE=1
# EMBEDDING_DIM=100                 # dimension of vectors
# EMBEDDING_PATH='/'                # file for pre-trained vectors
# EXPERIMENT_NAME='best_case'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=0 python -m se_tasks.code_authorship.scripts.main --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embedding_dim=$EMBEDDING_DIM --classes=$CLASSES --embedding_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embedding_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME | tee $EXPERIMENT_LOG



# EMBEDDING_TYPE=2
# EMBEDDING_DIM=100
# EMBEDDING_PATH='/'
# EXPERIMENT_NAME='worst_case'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=0 python -m se_tasks.code_authorship.scripts.main --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embedding_dim=$EMBEDDING_DIM --classes=$CLASSES --embedding_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embedding_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME | tee $EXPERIMENT_LOG



# EMBEDDING_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec100_1/doc2vec.vec'
# EXPERIMENT_NAME='100_1_doc2vec'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=0 python -m se_tasks.code_authorship.scripts.main --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embedding_dim=$EMBEDDING_DIM --classes=$CLASSES --embedding_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embedding_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME | tee $EXPERIMENT_LOG


# EMBEDDING_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec100_1/word2vec.vec'
# EXPERIMENT_NAME='100_1_word2vec'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=0 python -m se_tasks.code_authorship.scripts.main --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embedding_dim=$EMBEDDING_DIM --classes=$CLASSES --embedding_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embedding_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME | tee $EXPERIMENT_LOG


# EMBEDDING_TYPE=0
# EMBEDDING_DIM=100
# EMBEDDING_PATH='embedding_vec100_1/fasttext.vec'
# EXPERIMENT_NAME='100_1_fasttext'
# EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
# echo $EXPERIMENT_NAME
# CUDA_VISIBLE_DEVICES=0 python -m se_tasks.code_authorship.scripts.main --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
# --embedding_dim=$EMBEDDING_DIM --classes=$CLASSES --embedding_path=$EMBEDDING_PATH \
# --train_data=$TRAIN_DATA --test_data=$TEST_DATA --embedding_type=$EMBEDDING_TYPE \
# --experiment_name=$EXPERIMENT_NAME | tee $EXPERIMENT_LOG


EMBEDDING_TYPE=0
EMBEDDING_DIM=100
EMBEDDING_PATH='embedding_vec100_1/glove.vec'
EXPERIMENT_NAME='100_1_glove'
EXPERIMENT_LOG=$RES_DIR$EXPERIMENT_NAME'.txt'
echo $EXPERIMENT_NAME
CUDA_VISIBLE_DEVICES=0 python -m se_tasks.code_authorship.scripts.main --epochs=$EPOCHS --batch=$BATCH --lr=$LR \
--embedding_dim=$EMBEDDING_DIM --classes=$CLASSES --embedding_path=$EMBEDDING_PATH \
--train_data=$TRAIN_DATA --test_data=$TEST_DATA --embedding_type=$EMBEDDING_TYPE \
--experiment_name=$EXPERIMENT_NAME | tee $EXPERIMENT_LOG