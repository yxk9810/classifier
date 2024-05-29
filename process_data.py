#coding:utf-8
import sys 
import json 
def get_sentence(doc,event_info,is_start=False):
    end_dots = list("。!！？；;?")
    class_name = event_info['trigger'][0]['text']
    start = event_info['trigger'][0]['start']
    end = event_info['trigger'][0]['end']
    l = start
    while l>0 and doc[l] not in end_dots:
        l-=1
    r = end
    while r<len(doc) and doc[r] not in end_dots:
        r+=1
    before = doc[l+1:start]
    after = doc[end:r+1]
    if not is_start:
        return class_name+':'+before+"<t>"+doc[start:end]+"</t>"+after
    else:
        return class_name+':'+before+"<s>"+doc[start:end]+"</s>"+after
def load_json(filename):
    return json.load(open(filename,'r',encoding='utf-8'))
path='d:/round1_traning_data'
def convert_raw_to_has_relation(filename):
    dataList = load_json(filename)
    dataset = []
    for data in dataList:
        event_info = {}
        t_list = []
        for event in data['events']:
            event_info[event['id']] = event['event-information']
        uniq = set()
        for rel in data['relations']:
            one_id = rel['one_event_id']
            other_id = rel['other_event_id']
            key = one_id+'\t'+other_id
            uniq.add(key)
            first_event = event_info[one_id]
            second_event = event_info[other_id]
            json_data = {'text_a':get_sentence(data['doc'],first_event,is_start=True),'text_b':get_sentence(data['doc'],second_event),'label':1}
            t_list.append(json_data)
        
        n = len(data['events'])
        for i in range(n):
            for j in range(i+1,n):
                first = data['events'][i]
                second = data['events'][j]
                if first['id']+'\t'+second['id'] in uniq:continue
                json_data = {'text_a':get_sentence(data['doc'],first['event-information'],is_start=True),'text_b':get_sentence(data['doc'],second['event-information']),'label':0,'first':first['id'],'second':second['id']}
                t_list.append(json_data)
        dataset.extend(t_list)
    return dataset
train_file = path+'/train_raw.json'
dev_file = path+'/dev_raw.json'
test_file = path+'/test_raw.json'
train_data = convert_raw_to_has_relation(train_file)
dev_data = convert_raw_to_has_relation(dev_file)
test_data = convert_raw_to_has_relation(test_file)
def save(path,data):
    json.dump(data,open(path,'w',encoding='utf-8'),ensure_ascii=False)
save(path+'/train_has_rel.json',train_data)
save(path+'/dev_has_rel.json',dev_data)
save(path+'/test_has_rel.json',test_data)
