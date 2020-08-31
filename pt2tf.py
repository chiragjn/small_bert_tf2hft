import os
import sys
import json
from transformers import BertModel, TFBertModel


if __name__ == '__main__':
    path = sys.argv[1]    
    print(path)
    conf_path = os.path.join(path, 'config.json')
    c = json.load(open(conf_path))
    c['model_type'] = 'bert'
    json.dump(c, open(conf_path, 'w'))
    model = TFBertModel.from_pretrained(path, from_pt=True)
    model.save_pretrained(path)
    # test loads
    model = BertModel.from_pretrained(path)
    model = TFBertModel.from_pretrained(path)
