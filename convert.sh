#!/bin/bash
mkdir -p small_bert_checkpoints
mkdir -p hf_models

cd small_bert_checkpoints/
wget https://storage.googleapis.com/bert_models/2020_02_20/all_bert_models.zip
unzip all_bert_models.zip
find . -name 'uncased*.zip' -exec sh -c 'unzip -d "${1%.*}" "$1"' _ {} \;
rm uncased*.zip

for BERT_BASE_DIR in uncased*/ ; do
    transformers-cli convert --model_type bert   --tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt   --config $BERT_BASE_DIR/bert_config.json   --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin;
    mv $BERT_BASE_DIR/bert_config.json $BERT_BASE_DIR/config.json;
    rm $BERT_BASE_DIR/bert_model.ckpt*;
    python ../pt2tf.py $BERT_BASE_DIR
    mv $BERT_BASE_DIR "../hf_models/small_bert_${BERT_BASE_DIR}"
done

rm all_bert_models.zip
cd ../
python generate_readmes.py
python test_parity_with_tfhub.py
