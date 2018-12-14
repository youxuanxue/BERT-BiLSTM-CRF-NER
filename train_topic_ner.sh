#!/usr/bin/env bash
#
# Copyright @2018 R&D, Zhihu Inc. (zhihu.com)
# Author: xuejiao <xuejiao@zhihu.com>
#

export BERT_BASE_DIR=/data1/xuejiao/data/bert/model/chinese_L-12_H-768_A-12
export NER_DIR=/data1/xuejiao/data/ner/topic_ner

function cecho {
  local code="\033["
  case "$1" in
    black  | bk) color="${code}0;30m";;
    red    |  r) color="${code}1;31m";;
    green  |  g) color="${code}1;32m";;
    yellow |  y) color="${code}1;33m";;
    blue   |  b) color="${code}1;34m";;
    purple |  p) color="${code}1;35m";;
    cyan   |  c) color="${code}1;36m";;
    gray   | gr) color="${code}0;37m";;
    *) local text="$1"
  esac

  [ -z "${text}" ] && local text="${color}$2${code}0m"
  echo -e "${text}"
}

function usage() {
    cecho r "Usage: bash $0 action model_type [extra]"
    cecho r "action:"
    cecho r "      train | predict"
    cecho r "model_type:"
    cecho r "      raw | bert"
    exit 1
}

function args() {
    if [ $# -lt 1 ];then
        usage
    fi

    action=$1 && shift
    cecho b "action: ${action}"

    model_type=$1 && shift
    out_dir="${NER_DIR}/result/${model_type}_ner_output"
    if [ ! -d ${out_dir} ];then
        mkdir -p ${out_dir}
    fi
    cecho b "model_type: ${model_type}"

    PARAMS=$@
    cecho b "params: ${PARAMS}"
}


function train_raw() {
    # train on raw bilstm-crf
    python run_ner.py \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --label_vocab_file=$NER_DIR/label_vocab.txt \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
      --with_bert=False \
      --extra_embedding_file=$NER_DIR/extra_embedding.txt \
      --extra_embedding_dim=256 \
      --do_train=True \
      --train_file=$NER_DIR/train.txt \
      --do_eval=True \
      --eval_file=$NER_DIR/dev.txt \
      --max_seq_length=128 \
      --train_batch_size=32 \
      --learning_rate=2e-5 \
      --num_train_epochs=10.0 \
      --droupout_rate=0.9 \
      --output_dir=${out_dir}
}

function train_bert() {
    # train on pre-trained bert
    python run_ner.py \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --label_vocab_file=$NER_DIR/label_vocab.txt \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
      --with_bert=True \
      --do_train=True \
      --train_file=$NER_DIR/train.txt \
      --do_eval=True \
      --eval_file=$NER_DIR/dev.txt \
      --max_seq_length=128 \
      --train_batch_size=32 \
      --learning_rate=2e-5 \
      --num_train_epochs=10.0 \
      --droupout_rate=0.9 \
      --output_dir=${out_dir}
}

function predict_raw() {
    # predict on raw bilstm-crf
    python run_ner.py \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --label_vocab_file=$NER_DIR/label_vocab.txt \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
      --with_bert=False \
      --extra_embedding_file=$NER_DIR/extra_embedding.txt \
      --extra_embedding_dim=256 \
      --do_predict=True \
      --predict_file=$NER_DIR/dev.txt \
      --max_seq_length=128 \
      --droupout_rate=0.9 \
      --output_dir=${out_dir}
}

function predict_bert() {
    # predict on bert bilstm-crf
    python run_ner.py \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --label_vocab_file=$NER_DIR/label_vocab.txt \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
      --with_bert=True \
      --do_predict=True \
      --predict_file=$NER_DIR/dev.txt \
      --max_seq_length=128 \
      --droupout_rate=0.9 \
      --output_dir=${out_dir}
}

args $@

case "${action}_${model_type}" in
    "train_raw")
        train_raw;;
     "train_bert")
        train_bert;;
     "predict_raw")
        predict_raw;;
     "predict_bert")
        predict_bert;;
     *)
     usage;;
esac
