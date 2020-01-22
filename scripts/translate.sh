cd ..

PARAM_SET=tiny
MODEL_DIR=$PWD/transformer/model_$PARAM_SET
VOCAB_FILE=$PWD/resource/vocab.txt

FILE=$PWD/data/dev.txt
FILE_OUT=$PWD/data/dev.out.txt

python3 translate.py --model_dir=$MODEL_DIR \
    --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET \
    --file=$FILE --file_out=$FILE_OUT
