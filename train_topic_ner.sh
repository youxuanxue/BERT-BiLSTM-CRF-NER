#!/usr/bin/env bash
# @author xuejiao

export BERT_BASE_DIR=/data1/xuejiao/data/bert/model/chinese_L-12_H-768_A-12
export NER_DIR=/data1/xuejiao/data/ner/topic_ner

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
  --output_dir=$NER_DIR/result/raw_ner_output/


# train on pre-trained bert
python run_ner.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --label_vocab_file=$NER_DIR/label_vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$NER_DIR/train.txt \
  --do_eval=True \
  --eval_file=$NER_DIR/dev.txt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=10.0 \
  --droupout_rate=0.9 \
  --output_dir=$NER_DIR/result/bert_ner_output/

