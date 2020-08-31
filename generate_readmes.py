import re
import glob

README_TEMPLATE = """
{model_name}
---

This model was converted from the [release](https://github.com/google-research/bert/blob/8028c0459485299fa1ae6692b2300922a3fa2bad/README.md) of 24 smaller BERT models (English only, uncased, trained with WordPiece masking) referenced in [Well-Read Students Learn Better: On the Importance of Pre-training Compact Models.](https://arxiv.org/abs/1908.08962)

Conversion was performed automatically using `transformers-cli convert` as explained [here](https://huggingface.co/transformers/converting_tensorflow_models.html)

This model is also available on tfhub at `https://tfhub.dev/google/small_bert/{hub_handle}/1`

A small test was performed to check if converted model and the version at TFHub generate similar embeddings.  
https://github.com/chiragjn/small_bert_tf2hft/
"""

def main():
    for model_dir in glob.glob('hf_models/small_bert_*'):
        model_name = model_dir.split('/')[-1]
        hub_handle = model_name.split('small_', 1)[-1]
        contents = README_TEMPLATE.format(model_name=model_name, hub_handle=hub_handle).strip()
        with open(model_dir + '/README.md', 'w') as f:
            f.write(contents)

if __name__ == '__main__':
    main()
