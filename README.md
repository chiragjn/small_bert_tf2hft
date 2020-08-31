Quick and dirty scripts to convert TF checkpoints [released](https://github.com/google-research/bert/blob/8028c0459485299fa1ae6692b2300922a3fa2bad/README.md) of 24 smaller BERT models (English only, uncased, trained with WordPiece masking) referenced in [Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/abs/1908.08962) to huggingface/transformers format.

All converted models are available at https://huggingface.co/chiragjn

---

`convert.sh` does all the work for all 24 models.  

Converted Models are supposed to be equivalent to their [tfhub versions](https://tfhub.dev/s?q=small_bert)

**Bugs and Todo:**  
  - Models with hidden size 768 i.e. `small_bert_uncased_L-*_H-768_A-*` generate different embeddings after conversion from their tfhub counterparts

---
Code in the `bert/` folder was (selectively and shamelessly) copy pasted from https://github.com/google-research/bert/tree/eedf5716ce1268e56f0a50264a88cafad334ac61 for testing differences in conversions.

All kudos goes to authors of `google-research/bert` for making the pretrained checkpoints available and `huggingface/transformers` for making the conversion process painless.
