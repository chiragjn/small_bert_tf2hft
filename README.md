**Please note that the official converted models are already available at https://huggingface.co/google. It is recommended to use them instead of models from this repo**

I didn't realise this until later, so just consider this repo as exercise in converting models.

---
Quick and dirty scripts to convert [released TF checkpoints](https://github.com/google-research/bert/blob/8028c0459485299fa1ae6692b2300922a3fa2bad/README.md) of 24 smaller BERT models (English only, uncased, trained with WordPiece masking) referenced in [Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/abs/1908.08962) to huggingface/transformers format.

All converted models were previously available (deleted on Dec 30, 2020) at https://huggingface.co/chiragjn

---

`convert.sh` does all the work for all 24 models. (needs packages from requirements.txt)
`test_parity_with_tfhub.py` checks parity between converted models and tfhub versions (needs packages from requirements.txt)
`test_parity_with_ckpt.py` checks parity between tf checkpoints and tfhub versions (needs packages from requirements.ckptparity.txt)

---

Converted Models are equivalent to their released checkpoints - verified.  
Converted Models are supposed to be equivalent to their [tfhub versions](https://tfhub.dev/s?q=small_bert) - only 18/24 are equivalent.

Both original tf checkpoints and converted models with hidden size 768 i.e. `small_bert_uncased_L-*_H-768_A-12` generate different embeddings from their tfhub counterparts. This was been [reported](https://github.com/tensorflow/hub/issues/661) to TFHub team. **TF team published v2 for all models to tfhub.dev that resolves this**

**Todo:**  
  - Add code for testing parity between tf checkpoint and converted models. A bit tedious because of tf version problems. Needs tf 1.15 + transformers 3.x
  - Cleanup the repo, different requirements mess.

---
For testing equivalences in conversion, code in the `bert/` folder was (selectively and shamelessly) copy pasted from https://github.com/google-research/bert/tree/eedf5716ce1268e56f0a50264a88cafad334ac61 and only necessary parts were kept.

All kudos goes to authors of `google-research/bert` for making the pretrained checkpoints available and `huggingface/transformers` for making the conversion process painless.
