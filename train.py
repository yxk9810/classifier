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
parser.add_argument("--train_file",default="train_data",type=str,help="")
parser.add_argument("--dev_file",default="dev_data",type=str,help="")
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
    train_file = args.train_file
    dev_file = args.dev_file 
    test_file = args.test_file
    target_dir = './models/'
    use_fgm = False
    use_cls = args.pooling=='cls'

import time
now_time = time.strftime("%Y%m%d%H", time.localtime())
from transformers import AdamW
config = Config()
def train(model, train_data_loader,device,optimizer,fgm=None):
    model.train()
    total_loss, total_accuracy = 0, 0
    for step, batch in enumerate(tqdm(train_data_loader)):
        sent_id, mask, like_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        model.zero_grad()
        logits_like = model(sent_id, mask)
        loss_fn =  nn.CrossEntropyLoss()
        loss = loss_fn(logits_like, like_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_item = loss.item()
        total_loss += loss_item
    avg_loss = total_loss / len(train_data_loader)
    return avg_loss

import numpy as np
from sklearn.metrics import f1_score
def evaluate(model, dev_data_loader,device):
    model.eval()
    total_loss, total_accuracy = 0, 0
    gold_like = []
    pred_like = []
    for step, batch in enumerate(dev_data_loader):
        sent_id, mask, like_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        logits_like = model(sent_id, mask)
        loss_fn =  nn.CrossEntropyLoss()
        loss = loss_fn(logits_like, like_labels)
        loss_item = loss.item()
        preds =torch.argmax(torch.softmax(logits_like,dim=-1),dim=-1).detach().cpu().numpy()
        gold = batch[2].detach().cpu().numpy()
        gold_like.extend(gold.tolist())
        pred_like.extend(preds.tolist())
        total_loss += loss_item
    avg_loss = total_loss/len(pred_like)
    from sklearn.metrics import classification_report
    golds = [int(d) for d in gold_like]
    preds = [int(p) for p in pred_like]
    print(classification_report(golds,preds))
    accuracy = f1_score(golds,preds)
    return avg_loss, accuracy


def test(model, dev_data_loader):
    model.eval()
    gold_like = []
    pred_like = []
    with torch.no_grad():
        for step, batch in enumerate(dev_data_loader):
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

dataset = NLPCCTaskDataSet(filepath=config.train_file,mini_test=False)
train_data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn = partial(collate_fn_nlpcc,tokenizer=tokenizer), shuffle=True)
dev_dataset = NLPCCTaskDataSet(filepath=config.dev_file,mini_test=False,is_test=False)
dev_data_loader =  DataLoader(dev_dataset, batch_size=4, collate_fn = partial(collate_fn_nlpcc,tokenizer=tokenizer), shuffle=False)

best_valid_loss = float('inf')
best_f1 =0.0
for epoch in range(config.epoch):
    print('\n Epoch {:} / {:}'.format(epoch+1 ,config.epoch ))
    train_loss = train(model,train_data_loader,device,optimizer)
    dev_loss,dev_f1 = evaluate(model,dev_data_loader,device)
    if dev_loss<best_valid_loss:
        best_f1 =dev_f1
        print('best f1 = '+str(best_f1))
        best_valid_loss = dev_loss
        torch.save(model.state_dict(), 'model_weights.pth')
    print('train loss {}'.format(train_loss))
    print('val loss {} val acc {}'.format(dev_loss,dev_f1))
model.load_state_dict(torch.load('model_weights.pth'))

test_dataset = NLPCCTaskDataSet(filepath=config.test_file,mini_test=False,is_test=False)
test_data_loader =  DataLoader(test_dataset, batch_size=4, collate_fn = partial(collate_fn_nlpcc,tokenizer=tokenizer), shuffle=False)
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
