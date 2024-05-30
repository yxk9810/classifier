#coding:utf-8
import sys 
import json 

import sys 
import json 
import collections 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--has_rel_pred", default="test_pred_bert", type=str, help="")
parser.add_argument("--rel_pred", default="/kaggle/output/", type=str, help="")
args = parser.parse_args()

id2pairs =collections.defaultdict(set)
with open(args.has_rel_pred,'r',encoding='utf-8') as lines:
    for line in lines:
        data = json.loads(line.strip())
        id=data['first'].split('_')[0]
        key =data['first']+data['second']
        if int(data['pred'])==1:
            id2pairs[id].add(key)
id2relations = collections.defaultdict(list)
with open(args.rel_pred,'r',encoding='utf-8') as lines:
    for line in lines:
        data = json.loads(line.strip())
        id = data['first'].split('_')[0]
        if data['first'] == data['second']:continue
        key = data['first']+data['second']
        if key not in id2pairs[id]:continue 
        id2relations[id].append((data['first'],data['second'],'因果' if int(data['pred'])==0 else '时序'))

golds = json.load(open('/kaggle/input/cks-ere/test_raw.json','r',encoding='utf-8'))

found_cnt =0 
total_cnt = 0
right = 0 
not_found = 0 
for d in golds:
    relations ={rel['one_event_id']+rel['other_event_id']:rel['relation_type'] for rel in d['relations']}
    total_cnt+=len(relations)
    preds = id2relations[d['new-ID']] if d['new-ID'] in id2relations else []
    
    if len(preds)>0:
        for p in preds:
            key = p[0]+p[1]
            if key in relations and relations[key]==p[-1]:
                right+=1
    else:
        not_found+=1
    found_cnt+=len(preds)
    
precision = float(right)/found_cnt 
recall = float(right)/total_cnt 
print(precision)
print(recall)
print(2*precision*recall/(precision+recall))