import os
import json
import glob

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from bert.modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint
from bert.run_classifier import InputExample, InputFeatures, convert_examples_to_features
from bert import tokenization

training = False

class AbstractBase(object):
    def get_outputs(self, text_as, text_bs=None):
        text_bs = text_bs or ([None] * len(text_as))
        ies = [InputExample(str(i), text_a, text_b) for i, (text_a, text_b) in enumerate(zip(text_as, text_bs))]
        inp_fs = convert_examples_to_features(examples=ies, label_list=[None],
                                              max_seq_length=self.max_seq_length, tokenizer=self.tok)
        input_ids = [inp_f.input_ids for inp_f in inp_fs]
        input_mask = [inp_f.input_mask for inp_f in inp_fs]
        segment_ids = [inp_f.segment_ids for inp_f in inp_fs]
        
        bert_inputs = {
            self.input_ids: input_ids,
            self.input_mask: input_mask,
            self.segment_ids: segment_ids,
        }
        bo = self.sess.run(self.bert_outputs, feed_dict=bert_inputs)
        sequence_output = bo['sequence_output']
        pooled_output = bo['pooled_output']
        return input_ids, input_mask, segment_ids, sequence_output, pooled_output


class TFHubSmallBERT(AbstractBase):
    def __init__(self, handle, training=False, max_seq_length=512):
        self.max_seq_length = max_seq_length
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.bert_module = hub.Module(handle, trainable=False, tags={'train'} if training else None)
            self.sess = tf.Session()
            self.sess.run(tf.group(tf.global_variables_initializer(), tf.tables_initializer()))
            tokenization_info = self.bert_module(signature='tokenization_info', as_dict=True)
            vocab_file, do_lower_case = self.sess.run([tokenization_info['vocab_file'], tokenization_info['do_lower_case']])
            self.input_ids = tf.placeholder(tf.int32, shape=(None, self.max_seq_length))
            self.input_mask = tf.placeholder(tf.int32, shape=(None, self.max_seq_length))
            self.segment_ids = tf.placeholder(tf.int32, shape=(None, self.max_seq_length))
            bert_inputs = dict(
                input_ids=self.input_ids,
                input_mask=self.input_mask,
                segment_ids=self.segment_ids,
            )
            self.bert_outputs = self.bert_module(bert_inputs, signature="tokens", as_dict=True)
        self.tok = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        
class CheckpointSmallBERT(AbstractBase):
    def __init__(self, path, training=False, max_seq_length=512):
        self.max_seq_length = max_seq_length
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_ids = tf.compat.v1.placeholder(tf.int32, shape=(None, self.max_seq_length))
            self.input_mask = tf.compat.v1.placeholder(tf.int32, shape=(None, self.max_seq_length))
            self.segment_ids = tf.compat.v1.placeholder(tf.int32, shape=(None, self.max_seq_length))
            self.bert_config = BertConfig.from_json_file(path + '/bert_config.json')
            self.bert_module = BertModel(config=self.bert_config, is_training=training,
                                         input_ids=self.input_ids, input_mask=self.input_mask, 
                                         token_type_ids=self.segment_ids, use_one_hot_embeddings=False)
            assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(
                tf.trainable_variables(), 
                path + '/bert_model.ckpt'
            )
            tf.train.init_from_checkpoint(path + '/bert_model.ckpt', assignment_map)
            self.sess = tf.compat.v1.Session()
            self.sess.run(tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()))
            self.bert_outputs = {
                'sequence_output': self.bert_module.get_sequence_output(),
                'pooled_output': self.bert_module.get_pooled_output(),
            }
            self.tok = tokenization.FullTokenizer(vocab_file=path + '/vocab.txt', do_lower_case=True)


def test(hub_handle, path):
    print('=' * 120)
    print(hub_handle, path)
    text_a = ['well read students', 'this is a model compression test']
    text_b = ['learn better', 'all okay?']
    msl = json.load(open(path + '/bert_config.json'))['max_position_embeddings']
    checkpoint_model = CheckpointSmallBERT(path, training=False, max_seq_length=msl)
    hub_model = TFHubSmallBERT(f'https://tfhub.dev/google/{hub_handle}/1', training=False, max_seq_length=msl)
    chiids, chim, chsids, chso, chpo = checkpoint_model.get_outputs(text_a, text_b)
    tfiids, tfim, tfsids, tfso, tfpo = hub_model.get_outputs(text_a, text_b)
    assert np.allclose(chso, tfso, atol=0.05), np.max(np.abs(tfso - chso))
    assert np.allclose(chpo, tfpo, atol=0.05), np.max(np.abs(tfpo - chpo))
    
    # some force cleanup
    del checkpoint_model
    del hub_model


if __name__ == '__main__':
    f = open('ckpt_test_report.txt', 'w')
    for chkpoint in glob.glob('small_bert_checkpoints/uncased_*'):
        try:
            hub_handle = 'small_bert/bert_' + chkpoint.split('/', 1)[-1]
            print('Testing', hub_handle, 'with', chkpoint, file=f, flush=True)
            test(hub_handle, chkpoint)
            print('OK', file=f, flush=True)
        except AssertionError as e:
            print(e, file=f, flush=True)
    f.close()
