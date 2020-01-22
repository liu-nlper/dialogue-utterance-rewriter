cd ..

PARAM_SET=tiny
DATA_TRAIN=$PWD/data/train.txt
DATA_DEV=$PWD/data/dev.txt
MODEL_DIR=$PWD/transformer/model_$PARAM_SET
VOCAB_FILE=$PWD/resource/vocab.txt
BERT_CHECKPOINT=None

BLEU_SOURCE=$DATA_DEV
BLEU_REF=$PWD/data/BLEU_REF.txt

TRAIN_EPOCHS=50

python3 transformer_main.py --data_train=$DATA_TRAIN \
    --data_dev=$DATA_DEV --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE \
    --param_set=$PARAM_SET --bleu_source=$BLEU_SOURCE --bleu_ref=$BLEU_REF \
    --bert_checkpoint=$BERT_CHECKPOINT --train_epochs=$TRAIN_EPOCHS
