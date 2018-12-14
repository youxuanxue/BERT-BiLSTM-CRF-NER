# !/usr/bin/python
# -*- coding: utf-8 -*-
import codecs
import collections
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

import lstm_crf_layer
from bert import modeling
from bert import optimization
from bert import tokenization

__author__ = 'xuejiao'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("label_vocab_file", None,
                    "The vocabulary file for the NER label.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

# lstm params
flags.DEFINE_integer('lstm_size', 128, 'size of lstm units')
flags.DEFINE_integer('num_layers', 1, 'number of rnn layers, default is 1')
flags.DEFINE_string('cell', 'lstm', 'which rnn cell used')

flags.DEFINE_float('droupout_rate', 0.5, 'Dropout rate')
flags.DEFINE_float('clip', 5, 'Gradient clip')

flags.DEFINE_bool('with_bert', True, "Whether using pre-trained bert")

flags.DEFINE_string("extra_embedding_file", None,
                    "Extra embedding word2vec file when training without bert")
flags.DEFINE_integer("extra_embedding_dim", 256, "Embedding dim for extra embedding file")

## Other parameters
flags.DEFINE_string("train_file", None, "")

flags.DEFINE_string("eval_file", None, "")

flags.DEFINE_string("predict_file", None, "")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 labels,
                 label_ids):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labels = labels
        self.label_ids = label_ids


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        if self.num_features % 10000 == 0:
            tf.logging.info("process feature %s", self.num_features)

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        if self.is_training:
            features["label_ids"] = create_int_feature(feature.label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def parse_file_len(fname):
    i = -1
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def read_ner_features(file_name, text_tokenizer, label_tokenizer, max_seq_length, is_training,
                      output_fn):
    """Loads a data file into tf record."""

    with tf.gfile.Open(file_name, 'r') as reader:
        for line in reader.readlines():
            cur_data = json.loads(line)

            tokens = []
            segment_ids = []
            labels = []
            label_ids = []

            tokens.append("[CLS]")
            segment_ids.append(0)
            labels.append("O")
            label_ids.append(0)

            for t in cur_data["chars"][0: max_seq_length - 1]:
                tokens.append(t)
                segment_ids.append(0)

            input_ids = text_tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            if is_training:
                # no label for CLS
                for l in cur_data["labels"][0: max_seq_length - 1]:
                    labels.append(l)
                while len(labels) < max_seq_length:
                    labels.append("O")
                label_ids = label_tokenizer.convert_tokens_to_ids(labels)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            feature = InputFeatures(tokens, input_ids, input_mask, segment_ids, labels, label_ids)

            # Run callback
            output_fn(feature)


def input_fn_builder(input_file, seq_length, repeat, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    if is_training:
        name_to_features["label_ids"] = tf.FixedLenFeature([seq_length], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if repeat:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def create_model(bert_config, with_bert, extra_embedding, is_training, is_prediction, input_ids,
                 input_mask, segment_ids, labels, num_labels, sequence_lengths, use_one_hot_embeddings):
    if with_bert:
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
        embedding = model.get_sequence_output()
        max_seq_length = embedding.shape[1].value

    else:
        tf.logging.info("")
        max_seq_length = FLAGS.max_seq_length
        extra_lookup = tf.get_variable(
            name="extra_lookup",
            shape=[extra_embedding.shape[0], extra_embedding.shape[1]])
        extra_lookup.assign(extra_embedding, True)

        embedding = tf.nn.embedding_lookup(extra_lookup, input_ids)

    blstm_crf = lstm_crf_layer.BLSTM_CRF(embedded_chars=embedding,
                                         hidden_unit=FLAGS.lstm_size,
                                         cell_type=FLAGS.cell,
                                         num_layers=FLAGS.num_layers,
                                         dropout_rate=FLAGS.droupout_rate,
                                         initializers=initializers,
                                         num_labels=num_labels,
                                         seq_length=max_seq_length,
                                         labels=labels,
                                         lengths=sequence_lengths,
                                         is_training=is_training,
                                         is_prediction=is_prediction)

    rst = blstm_crf.add_blstm_crf_layer(crf_only=False)
    return rst


def model_fn_builder(bert_config, with_bert, extra_embedding, num_labels, init_checkpoint,
                     learning_rate, num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        is_prediction = (mode == tf.estimator.ModeKeys.PREDICT)

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        if is_prediction:
            label_ids = None
        else:
            label_ids = features["label_ids"]

        used = tf.sign(tf.abs(input_ids))
        length = tf.reduce_sum(used, reduction_indices=1)
        sequence_lengths = tf.cast(length, tf.int32)

        (logits, trans, total_loss, pred_ids) = create_model(
            bert_config=bert_config,
            with_bert=with_bert,
            extra_embedding=extra_embedding,
            is_training=is_training,
            is_prediction=is_prediction,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            labels=label_ids,
            num_labels=num_labels,
            sequence_lengths=sequence_lengths,
            use_one_hot_embeddings=use_one_hot_embeddings)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids, predict_labels):
                accuracy = tf.metrics.accuracy(label_ids, predict_labels)
                precision = tf.metrics.precision(label_ids, predict_labels)
                recall = tf.metrics.recall(label_ids, predict_labels)
                return {
                    "eval_accuracy": accuracy,
                    "eval_precision": precision,
                    "eval_recall": recall,
                }

            eval_metrics = (metric_fn, [label_ids, pred_ids])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)

        else:
            if label_ids:
                predictions = {
                    "input_ids": input_ids,
                    "label_ids": label_ids,
                    "predict_ids": pred_ids,
                    "lengths": sequence_lengths,
                }
            else:
                predictions = {
                    "input_ids": input_ids,
                    "predict_ids": pred_ids,
                    "lengths": sequence_lengths,
                }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def load_embedding(path, word_dim, char_to_id):
    data = np.random.uniform(-1, 1, [len(char_to_id), word_dim])
    count = 0
    try:
        for i, line in enumerate(codecs.open(path, 'r', 'utf-8')):
            line = line.rstrip().split()
            if len(line) == word_dim + 1:
                if line[0] in char_to_id:
                    count += 1
                    id = char_to_id[line[0]]
                    data[id] = np.array([float(x) for x in line[1:]]).astype(np.float32)
                else:
                    pass
    except Exception as ex:
        tf.logging.error("load_embedding error", ex)
    tf.logging.info("load %d extra embedding for total chars: %d", count, len(char_to_id))
    return data


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    text_tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    extra_embedding = None
    if FLAGS.extra_embedding_file:
        extra_embedding = load_embedding(
            FLAGS.extra_embedding_file, FLAGS.extra_embedding_dim, text_tokenizer.vocab)

    label_tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.label_vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    num_train_steps = None
    num_warmup_steps = None
    example_count = 0

    if FLAGS.do_train:
        example_count = parse_file_len(FLAGS.train_file)
        num_train_steps = int(example_count / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    elif FLAGS.do_eval:
        example_count = parse_file_len(FLAGS.eval_file)
    elif FLAGS.do_predict:
        example_count = parse_file_len(FLAGS.predict_file)
    else:
        pass

    model_fn = model_fn_builder(
        bert_config=bert_config,
        with_bert=FLAGS.with_bert,
        extra_embedding=extra_embedding,
        num_labels=len(label_tokenizer.vocab),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu
    )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        # We write to a temporary file to avoid storing very large constant tensors
        # in memory.
        train_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
            is_training=True)
        read_ner_features(
            file_name=FLAGS.train_file,
            text_tokenizer=text_tokenizer,
            label_tokenizer=label_tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            is_training=True,
            output_fn=train_writer.process_feature)
        train_writer.close()

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num orig examples = %d", example_count)
        tf.logging.info("  Num split examples = %d", train_writer.num_features)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        train_input_fn = input_fn_builder(
            input_file=train_writer.filename,
            seq_length=FLAGS.max_seq_length,
            repeat=True,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        # We write to a temporary file to avoid storing very large constant tensors
        # in memory.
        eval_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
            is_training=True)
        read_ner_features(
            file_name=FLAGS.eval_file,
            text_tokenizer=text_tokenizer,
            label_tokenizer=label_tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            is_training=True,
            output_fn=eval_writer.process_feature)
        eval_writer.close()

        eval_steps = int(example_count / FLAGS.eval_batch_size)

        tf.logging.info("***** Running evaluating *****")
        tf.logging.info("  Num orig examples = %d", example_count)
        tf.logging.info("  Num split examples = %d", eval_writer.num_features)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        tf.logging.info("  Num steps = %d", eval_steps)

        # eval
        eval_input_fn = input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=FLAGS.max_seq_length,
            repeat=False,
            is_training=True,
            drop_remainder=True)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        # eval report
        report_input_fn = input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=FLAGS.max_seq_length,
            repeat=False,
            is_training=True,
            drop_remainder=True)

        def build_result(input_ids, label_ids, predict_ids, lengths):
            predict_results = []
            chars = [text_tokenizer.inv_vocab[x] for x in input_ids[:lengths]]
            golds = [label_tokenizer.inv_vocab[x] for x in label_ids[:lengths]]
            preds = [label_tokenizer.inv_vocab[x] for x in predict_ids[:lengths]]
            for char, gold, pred in zip(chars, golds, preds):
                if type(char) is bytes:
                    char = char.decode()
                predict_results.append(" ".join([char, gold, pred]))

            return predict_results

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        detail_out_file = os.path.join(FLAGS.output_dir, "eval_result_detail.txt")
        tf.logging.info("Writing eval detail to: %s" % (detail_out_file))

        with tf.gfile.GFile(detail_out_file, "w") as writer:
            count = 0
            for result in estimator.predict(report_input_fn, yield_single_examples=True):
                count += 1
                if count % 1000 == 0:
                    tf.logging.info("Processing example: %d" % count)
                predict_block = build_result(
                    result["input_ids"],
                    result["label_ids"],
                    result["predict_ids"],
                    result["lengths"])
                for line in predict_block:
                    writer.write(line + "\n")
                writer.write("\n")

        from conlleval import return_report
        eval_report = return_report(detail_out_file)
        report_out_file = os.path.join(FLAGS.output_dir, "eval_result_report.txt")
        with tf.gfile.GFile(report_out_file, "w") as writer:
            for line in eval_report:
                writer.write(line + "\n")
                tf.logging.info(line)

    if FLAGS.do_predict:
        example_count = parse_file_len(FLAGS.predict_file)
        eval_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "pred.tf_record"),
            is_training=False)
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        read_ner_features(
            file_name=FLAGS.predict_file,
            text_tokenizer=text_tokenizer,
            label_tokenizer=label_tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            is_training=False,
            output_fn=append_feature)
        eval_writer.close()

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", example_count)
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=FLAGS.max_seq_length,
            repeat=False,
            is_training=False,
            drop_remainder=False)

        def build_result(input_ids, predict_ids, lengths):
            task_result = []
            cur_token = []
            chars = [text_tokenizer.inv_vocab[x] for x in input_ids[:lengths]]
            preds = [label_tokenizer.inv_vocab[x] for x in predict_ids[:lengths]]
            for char, pred in zip(chars, preds):
                if pred in ["O"]:
                    cur_token.append(char)
                    task_result.append("".join(cur_token))
                    cur_token = []
                else:
                    pieces = pred.split('-')
                    if len(pieces) == 2:
                        if pieces[0] in ["E"]:
                            cur_token.append(char + "/" + pieces[1])
                            task_result.append("".join(cur_token))
                            cur_token = []
                        else:
                            cur_token.append(char)
                    else:
                        cur_token.append(char)
            if cur_token:
                task_result.append("".join(cur_token))
            return "".join(task_result)

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        detail_out_file = os.path.join(FLAGS.output_dir, "predictions.txt")
        tf.logging.info("Writing predictions to: %s" % (detail_out_file))

        with tf.gfile.GFile(detail_out_file, "w") as writer:
            count = 0
            for result in estimator.predict(predict_input_fn, yield_single_examples=True):
                count += 1
                if count % 1000 == 0:
                    tf.logging.info("Processing example: %d" % count)
                info = build_result(result["input_ids"], result["predict_ids"], result["lengths"])
                writer.write("%s\n" % info)


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("label_vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
