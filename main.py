# import pandas as pd
# import numpy as np
# import pickle as pk

# import os
# import time
# import copy
# import torch
# import torch.nn as nn
# from transformers import AutoTokenizer, LongformerForSequenceClassification,Trainer,AutoModel,LongformerTokenizerFast,RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments

# from train_utils import train_with_early_stopping, split_df_by_pt, dump_outputs, update_args, eval_test
# from eval_utils import full_metrics, result2str, reg_metrics, score_f1, score_mae
# from data_utils import DrgTextDataset, load_rule

# from options import args
# from models import pick_model


# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from tqdm import tqdm
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = args.device

# # def main():

# def tokenization(batched_text):
#     return tokenizer(batched_text['text'], padding = 'max_length', truncation=True, max_length = 1024)  
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
#     acc = accuracy_score(labels, preds)
#     return {
#         'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }
# if __name__ == '__main__': 
#     # main()
#     RULE_PATH = '%s/%sDRG_RULE%s.csv' % (args.rule_dir, args.cohort.upper(), args.rule)

#     data_dir = '%s/%s' % (args.data_dir, args.cohort)
#     TEXT_DIR = '%s/text_embed2' % data_dir
#     embedding = np.load('%s/embedding.npy' % data_dir) 

#     run_time = time.strftime('%b_%d_%H', time.localtime())
#     result_dir = 'results/%s' % '_'.join([args.cohort, args.model, 'text', run_time])
#     tokenizer = LongformerTokenizerFast.from_pretrained("EarlyDRGPrediction/clinicalLongformer",ignore_mismatched_sizes=True,max_length = 2000)


#     train_val_df = pd.read_csv('%s/train_val.csv' % data_dir)
#     test_df = pd.read_csv('%s/test.csv' % data_dir)

#     train_df, dev_df = split_df_by_pt(train_val_df, frac=0.1)
    
    
    
# #     dataset_file = "dataset.pkl"
# #     if os.path.exists(dataset_file):
# #         with open(dataset_file, "rb") as file:
# #             train_dataset = pk.load(file)
# #     if not args.eval_model:
#     train_dataset = DrgTextDataset(args, train_df, RULE_PATH)
#     dev_dataset = DrgTextDataset(args, dev_df, RULE_PATH)
#     test_dataset = DrgTextDataset(args, test_df, RULE_PATH)
#     args.Y = test_dataset.Y 
#     with open('train_dataset.pkl', 'wb') as file:
#         pk.dump(train_dataset, file)
#     with open('dev_dataset.pkl', 'wb') as file:
#         pk.dump(dev_dataset, file)
#     with open('test_dataset.pkl', 'wb') as file:
#         pk.dump(test_dataset, file)
        
        
        
#     model = LongformerForSequenceClassification.from_pretrained("EarlyDRGPrediction/clinicalLongformer",num_labels=args.Y,ignore_mismatched_sizes=True)

# #     train_data = train_data.map(tokenization, batched = True, batch_size = len(train_data))
# #     test_data = test_data.map(tokenization, batched = True, batch_size = len(test_data))
    
# #     training_args = TrainingArguments(
# #     output_dir = '/media/data_files/github/website_tutorials/results',
# #     num_train_epochs = 5,
# #     per_device_train_batch_size = 8,
# #     gradient_accumulation_steps = 8,    
# #     per_device_eval_batch_size= 16,
# #     evaluation_strategy = "epoch",
# #     disable_tqdm = False, 
# #     load_best_model_at_end=True,
# #     warmup_steps=200,
# #     weight_decay=0.01,
# #     logging_steps = 4,
# #     fp16 = True,
# # #     logging_dir='/media/data_files/github/website_tutorials/logs',
# #     dataloader_num_workers = 4,
# #     run_name = 'clinical-longformer-classification'
# #     )
# #     trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     compute_metrics=compute_metrics,
# #     train_dataset=train_data,
# #     eval_dataset=test_data
# #     )
# #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
# #     trainer.train()
# #     trainer.evaluate()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     if not args.eval_model:
# #         model = pick_model(args, embedding)

#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
#         if args.target == 'drg':
#             score_func = score_f1
#             small_base = True
# #         else:
# #             score_func = score_mae
# #             small_base = False
#         model_wts = train_with_early_stopping(model, train_dataset, dev_dataset, args.epochs, args.patience, args.target, args.batch_size, optimizer, score_func, small_base)
# #     else:
# #         print("load checkpoint from", args.eval_model)
# #         eval_path = args.eval_model
# #         hyperparam_path = '%s/hyperparam.pk' % args.eval_model
# #         if os.path.exists(hyperparam_path):
# #             hyperparam = pd.read_pickle(hyperparam_path)
# #             args_new = update_args(copy.deepcopy(args), hyperparam)
# #         model = pick_model(args_new, embedding)
# #         model_wts = torch.load('%s/checkpoint.bin' % args.eval_model)
# #         if args_new.target == 'drg':
# #             score_func = score_f1
# #             small_base = True
# #         else:
# #             score_func = score_mae
# #             small_base = False

# #     # eval 
# #     model.load_state_dict(model_wts)
# #     text_infs = {}
# #     for hour in [24, 48]:
# #         print('\nTest Hour', str(hour), 'Evaluation Results')
# #         test_dataset.load_data(hour)
# #         te_score, te_inf = eval_test(model, test_dataset, args.target, score_func, RULE_PATH, True, args.batch_size)
# #         text_infs['inf%s' % hour] = te_inf

# #     if args.save_model:
# #         dump_outputs(result_dir, text_infs, checkpoint=model_wts, hyperparam=vars(args))

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
    if args.cohort=='ms':
        RULE_PATH = 'rules/MSDRG_RULE13.csv' 
    else:
        RULE_PATH = 'rules/APRDRG_RULE13.csv' 
    data_dir = '%s/%s' % ('data',  args.cohort)
    TEXT_DIR = '%s/text_tokens' % data_dir

    embedding = np.load('%s/embeddingAug.npy' % TEXT_DIR) 
    print(embedding.shape)
    run_time = time.strftime('%b_%d_%H_%M', time.localtime())
    if args.LongFormer=='No':
        with open('test_dataset_%s.pkl'% args.cohort, "rb") as file: 
            test_dataset = pk.load(file)
    else:
        with open('test_dataset_%s.pkl'% args.cohort, "rb") as file: 
            test_dataset = pk.load(file)



    Atten=Attention(input_size=512,output_size=150) 
    model = model(args, embedding,Atten,0)    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    score_func = score_f1
    model=model.cuda()
    if args.eval_model == 'train':
        print('train')
        if args.LongFormer=='No':
            model = LongformerForSequenceClassification.from_pretrained("yikuan8/Clinical-Longformer",num_labels=738,ignore_mismatched_sizes=True)
            # num_layers_to_freeze=5
            # for param in model.longformer.encoder.layer[:num_layers_to_freeze].parameters():
            #     param.requires_grad = False
            with open('train_dataset_%s_LongFormer.pkl'% args.cohort, "rb") as file: 
                train_dataset = pk.load(file)
            with open('dev_dataset_%s_LongFormer.pkl'% args.cohort, "rb") as file: 
                dev_dataset = pk.load(file)
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-5)

        else:
            with open('train_dataset_%s_LongFormer.pkl'% args.cohort, "rb") as file: 
                train_dataset = pk.load(file)
            with open('dev_dataset_%s.pkl'% args.cohort, "rb") as file: 
                dev_dataset = pk.load(file)
            model_wts = train_dev(model,train_dataset, dev_dataset, args.epochs,  args.batch_size, optimizer, score_func)
            model_wts = train_dev(args,model, train_dataset, dev_dataset, 'drg', 4, optimizer, score_func,epochs=6, patience=3)
    else:
        print('eval')
        if args.LongFormer=='No':
            state_dict = torch.load("best%s" % args.cohort.upper())['model_state_dict']   
        else:
            state_dict = torch.load("best%s_LongFormer" % args.cohort.upper())['model_state_dict']   
        model.load_state_dict(state_dict)

    if args.LongFormer=='No':
        te_score, te_inf = eval_test(model,test_dataset, score_func, RULE_PATH, True, args.batch_size)
    else:
        te_score, te_inf = eval_test(model, test_dataset, score_func, RULE_PATH, True, 4)

