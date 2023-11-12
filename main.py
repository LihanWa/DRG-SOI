import pandas as pd
import numpy as np
import pickle as pk
import os
import time
import copy
import torch
import torch.nn as nn
from train_utils import train_dev, eval_test
from eval_utils import full_metrics, result2str, reg_metrics, score_f1, score_mae
from data_utils_ori_2000 import DrgTextDataset, load_rule
from models import Attention
from options import args
from models import model
from transformers import AutoTokenizer, LongformerForSequenceClassification,Trainer,AutoModel,LongformerTokenizerFast,RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments
os.environ['CUDA_VISIBLE_DEVICES'] = args.device



if __name__ == '__main__':
    if args.data_source=='ms':
        RULE_PATH = 'rules/MSDRG_RULE13.csv' 
    else:
        RULE_PATH = 'rules/APRDRG_RULE13.csv' 
    data_dir = '%s/%s' % ('data',  args.data_source)
    TEXT_DIR = '%s/text_tokens' % data_dir
    if args.LongFormer=='No':
      embedding = np.load('%s/embeddingAug.npy' % TEXT_DIR) 
      print(embedding.shape)
    run_time = time.strftime('%b_%d_%H_%M', time.localtime())
    if args.LongFormer=='No':
        with open('test_dataset_%s.pkl'% args.data_source, "rb") as file: 
            test_dataset = pk.load(file)
    else:
        with open('test_dataset_%s_LongFormer.pkl'% args.data_source, "rb") as file: 
            test_dataset = pk.load(file)
    score_func = score_f1

    if args.LongFormer=='Yes':
        if args.data_source=='ms':
            model = LongformerForSequenceClassification.from_pretrained("yikuan8/Clinical-Longformer",num_labels=738,ignore_mismatched_sizes=True)
        else:
            model = LongformerForSequenceClassification.from_pretrained("yikuan8/Clinical-Longformer",num_labels=1136,ignore_mismatched_sizes=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    else:
        Atten=Attention(input_size=512,output_size=150) 
        model = model(args, embedding,Atten,0)    
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        model=model.cuda()
    
    if args.eval_model == 'train':
        print('train')
        if args.LongFormer=='Yes':
            state_dict = torch.load("best%s_LongFormer" % args.data_source.upper())['model_state_dict']   
            model.load_state_dict(state_dict)

            # num_layers_to_freeze=5
            # for param in model.longformer.encoder.layer[:num_layers_to_freeze].parameters():
            #     param.requires_grad = False
            with open('dev_dataset_%s_LongFormer.pkl'% args.data_source, "rb") as file: 
                dev_dataset = pk.load(file)
            with open('train_dataset_%s_LongFormer.pkl'% args.data_source, "rb") as file: 
                train_dataset = pk.load(file)                
            
        else:
            with open('train_dataset_%s.pkl'% args.data_source, "rb") as file: 
                train_dataset = pk.load(file)
            with open('dev_dataset_%s.pkl'% args.data_source, "rb") as file: 
                dev_dataset = pk.load(file)
        model_wts = train_dev(model,train_dataset, dev_dataset, args.epochs,  args.batch_size, optimizer, score_func)
    else:
        print('eval')
        if args.LongFormer=='No':
            state_dict = torch.load("best%s" % args.data_source.upper())['model_state_dict']   
        else:
            state_dict = torch.load("best%s_LongFormer" % args.data_source.upper())['model_state_dict']   
        model.load_state_dict(state_dict)
        
    te_score, te_inf = eval_test(model,test_dataset, score_func, RULE_PATH, True, args.batch_size)
