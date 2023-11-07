import pandas as pd
import pickle as pk
from transformers import LongformerForSequenceClassification,LongformerTokenizerFast
from train_utils import split_df_by_pt_mimic3,split_df_by_pt_mimic4
import numpy as np
from options import args
from data_utils_ori_2000 import DrgTextDataset
import os
if args.data_source=='ms':
    RULE_PATH = 'rules/MSDRG_RULE13.csv' 
else:
    RULE_PATH = 'rules/APRDRG_RULE13.csv' 
data_dir = 'data/%s' % args.data_source.upper()
# aug_df=pd.read_csv('/Users/lihan/Downloads/EarlyDRGPrediction-main/5_aug.csv')
train_val_df = pd.read_csv('%s/train_val.csv' % data_dir)
test_df = pd.read_csv('%s/test.csv' % data_dir)
drg=pd.read_csv('%s/drg_cohort.csv' % data_dir)
if args.data_source=='apr_mimic4':
    train_df, dev_df = split_df_by_pt_mimic4(train_val_df, frac=0.1)
else:
    train_df, dev_df = split_df_by_pt_mimic3(train_val_df, frac=0.1)
aug_df=pd.read_csv('/Users/lihan/Downloads/EarlyDRGPrediction-main/5_aug.csv')
if  args.data_source=='ms' and args.LongFormer=='No':
    train_df = pd.concat([train_df, aug_df], ignore_index=True)
train_dataset = DrgTextDataset(args, train_df, RULE_PATH)


test_dataset = DrgTextDataset(args, test_df, RULE_PATH)
dev_dataset = DrgTextDataset(args, dev_df, RULE_PATH)
if args.LongFormer=='Yes':
    datasets=[train_dataset,test_dataset,dev_dataset]
    tokenizer = LongformerTokenizerFast.from_pretrained("yikuan8/Clinical-Longformer",ignore_mismatched_sizes=True,max_length = 2000)
    for i in datasets:
        for j,v in enumerate(i):
            # print(v['text'])
            # print(type([v['text'][0]]))
            # print(type(v['text'][0]))

            tokens = tokenizer(v['text'], return_tensors="pt",max_length=2000, truncation=True, padding = 'max_length')
            # tokens['input_ids']=tokens['input_ids']
            # tokens['attention_mask']=tokens['attention_mask']
            v['text']=tokens

            if j==len(i)-1: break
    with open('train_dataset_%s_LongFormer.pkl'% args.data_source, 'wb') as file:
        pk.dump(train_dataset, file)
    with open('dev_dataset_%s_LongFormer.pkl'% args.data_source, 'wb') as file:
        pk.dump(dev_dataset, file)
    with open('test_dataset_%s_LongFormer.pkl'% args.data_source, 'wb') as file:
        pk.dump(test_dataset, file)
    
else:
    with open('train_dataset_%s.pkl'% args.data_source, 'wb') as file:
        pk.dump(train_dataset, file)
    with open('dev_dataset_%s.pkl'% args.data_source, 'wb') as file:
        pk.dump(dev_dataset, file)
    with open('test_dataset_%s.pkl'% args.data_source, 'wb') as file:
        pk.dump(test_dataset, file)