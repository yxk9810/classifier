from transformers import BertModel
from transformers import AutoModel
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
from tqdm import tqdm
from loss import FocalLoss
from model import BertClassifier
from dataset import NLPCCTaskDataSet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", default="bert-base-chinese", type=str, help="")
parser.add_argument("--output_dir", default="/kaggle/output/", type=str, help="")
parser.add_argument("--output_filename", default="test_pred_bert", type=str, help="")
parser.add_argument("--test_file",default="",type=str,help="")
parser.add_argument("--device", default="cpu", type=str, help="")
parser.add_argument("--pooling", default="cls", type=str, help="")
parser.add_argument("--hidden_size", default=768, type=int, help="")
parser.add_argument("--seed", default=42, type=int, help="")
parser.add_argument("--batch_size", default=32, type=int, help="")
parser.add_argument("--epochs", default=5, type=int, help="")
parser.add_argument("--class_num",default=2,type=int,help="please inpu class num ")
parser.add_argument("--learning_rate", default=3e-5, type=float, help="")
args = parser.parse_args()
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if args.device=='cuda' else 'cpu')
class Config:
    pretrain_model_path = args.model_name_or_path
    hidden_size = args.hidden_size
    learning_rate = args.learning_rate
    class_num = args.class_num 
    epoch = args.epochs
    test_file = args.test_file
    target_dir = './models/'
    use_fgm = False
    use_cls = args.pooling=='cls'

import time
now_time = time.strftime("%Y%m%d%H", time.localtime())
from transformers import AdamW
config = Config()
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm 
def test(model, dev_data_loader):
    model.eval()
    gold_like = []
    pred_like = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dev_data_loader)):
            sent_id, mask, like_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            logits_like = model(sent_id, mask)
            preds =torch.argmax(torch.softmax(logits_like,dim=-1),dim=-1).detach().cpu().numpy()
            gold = batch[2].detach().cpu().numpy()
            gold_like.extend(gold.tolist())
            pred_like.extend(preds.tolist())
    return gold_like, pred_like

def collate_fn_nlpcc(batch, max_seq_lenght=256,tokenizer=None):
    batch_data = []
    batch_labels = []
    for d in batch:
        batch_data.append(d['text_a']+'[SEP]'+d['text_b'])
        batch_labels.append(int(d['label']) if 'label' in d else 0)
    tokens = tokenizer(
        batch_data,
        padding=True,
        max_length=max_seq_lenght,
        truncation=True)
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    labels = torch.tensor(batch_labels,dtype=torch.long)
    return seq, mask, labels


from functools import partial
from transformers import AutoTokenizer
model = BertClassifier(config)
optimizer = AdamW(model.parameters(), lr=config.learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(config.pretrain_model_path)
model.load_state_dict(torch.load('model_weights.pth'))
test_dataset = NLPCCTaskDataSet(filepath=config.test_file,mini_test=False,is_test=False)
test_data_loader =  DataLoader(test_dataset, batch_size=64, collate_fn = partial(collate_fn_nlpcc,tokenizer=tokenizer), shuffle=False)
golds,preds = test(model,test_data_loader)
from sklearn.metrics import classification_report
print(classification_report(golds,[int(p) for p in preds]))
import os
if os.path.exists(args.output_filename):
    os.remove(args.output_filename)
import json 
writer = open(args.output_filename,'a+',encoding='utf-8')
for pred,t in zip(preds,test_dataset.dataset):
    t['pred'] = pred 
    writer.write(json.dumps(t,ensure_ascii=False)+'\n')
writer.close()
