import os
import json
import glob

import numpy as np
import torch
import tensorflow as tf
import tensorflow_hub as hub
from transformers import AutoTokenizer, AutoModel
from bert.rc import InputExample, InputFeatures, convert_examples_to_features
from bert import tokenization

tf.compat.v1.disable_eager_execution()
training = False

class TFHubSmallBERT(object):
    def __init__(self, handle, training=False, max_seq_length=512):
        self.max_seq_length = max_seq_length
        with tf.compat.v1.Graph().as_default():
            self.bert_module = hub.Module(handle, trainable=False, tags={'train'} if training else None)
            self.sess = tf.compat.v1.Session()
            self.sess.run(tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()))
            tokenization_info = self.bert_module(signature='tokenization_info', as_dict=True)
            vocab_file, do_lower_case = self.sess.run([tokenization_info['vocab_file'], tokenization_info['do_lower_case']])
            self.input_ids = tf.compat.v1.placeholder(tf.int32, shape=(None, self.max_seq_length))
            self.input_mask = tf.compat.v1.placeholder(tf.int32, shape=(None, self.max_seq_length))
            self.segment_ids = tf.compat.v1.placeholder(tf.int32, shape=(None, self.max_seq_length))
            bert_inputs = dict(
                input_ids=self.input_ids,
                input_mask=self.input_mask,
                segment_ids=self.segment_ids,
            )
            self.bert_outputs = self.bert_module(bert_inputs, signature="tokens", as_dict=True)
        self.tok = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        
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


def test(hub_handle, path):
    text_a = ['well read students', 'this is a model compression test']
    text_b = ['learn better', 'all okay?']
    msl = json.load(open(path + '/' + 'config.json'))['max_position_embeddings']
    hub_model = TFHubSmallBERT(f'https://tfhub.dev/google/small_bert/{hub_handle}/1', training=False)
    tfiids, tfim, tfsids, tfso, tfpo = hub_model.get_outputs(text_a, text_b)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    hf_inputs = tokenizer(
        text=text_a,
        text_pair=text_b,
        max_length=msl,
        return_tensors='pt',
        padding='max_length',
        truncation=False,
    )
    model.eval()
    with torch.no_grad():
        to = model(**hf_inputs)
        
    ptiids = hf_inputs['input_ids'].detach().cpu().numpy().tolist()
    ptim = hf_inputs['attention_mask'].detach().cpu().numpy().tolist()
    ptsids = hf_inputs['token_type_ids'].detach().cpu().numpy().tolist()
    ptso, ptpo = to
    ptso = ptso.detach().cpu().numpy()
    ptpo = ptpo.detach().cpu().numpy()
    
    assert tfiids == ptiids
    assert tfim == ptim
    assert tfsids == ptsids
    assert np.allclose(tfso, ptso, atol=0.05), np.max(np.abs(tfso - ptso))
    assert np.allclose(tfpo, ptpo, atol=0.05), np.max(np.abs(tfpo - ptpo))
    


if __name__ == '__main__':
    f = open('test_report.txt', 'w')
    for hf_model in glob.glob('hf_models/small_*'):
        try:
            hub_handle = hf_model.split('hf_models/small_', 1)[-1]
            print('Testing', hub_handle, 'with', hf_model, file=f)
            test(hub_handle, hf_model)
            print('OK', file=f)
        except Exception as e:
            print(e, file=f)
    f.close()
