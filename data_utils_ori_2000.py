import os
import pandas as pd
import numpy as np
import pickle as pk
# from options import args

import torch
from torch.utils.data import Dataset


class DrgTextDataset(Dataset):
    def __init__(self, args, df, rule_path):
        # self.lessThan=lessThan

        self.df = df
        print('2000ab')
        self.max_seq_length = 2000
        self.data_dir = '%s/%s' % ('data', args.data_source)
        if args.LongFormer=='No':
            self.text_dir = '%s/text_tokens' % self.data_dir 
        else:
            print('a'*12)
            self.text_dir = '%s/text_LongFormer' % self.data_dir 
        # self.token2id_dir = '%s/token2id.dict' % self.data_dir
        # self.token2id = pd.read_pickle(self.token2id_dir)

        _, self.d2i, _, _, self.d2w = load_rule(rule_path)
        if args.data_source in ['ms','apr_mimic3']:
            print(args.data_source)
            self.unique_pt_df = self.df.sort_values(by=['SUBJECT_ID', 'hour0', 'hour12'], ascending=False).drop_duplicates(subset=['SUBJECT_ID']).reset_index(drop=True)
        else:
            self.unique_pt_df = self.df.sort_values(by=['subject_id', 'hour0', 'hour12'], ascending=False).drop_duplicates(subset=['subject_id']).reset_index(drop=True)
        self.load_data(args)
        print(len(self.d2i))

    def load_data(self,args):
        self.size = len(self.df)
        self.data = self.read_df(args,self.df)
        print('dataset loaded with', len(self.data), 'stays')

    def load_pop_for_hpa(self, hour):
        self.size = len(self.unique_pt_df)
        self.data = self.read_df(self.unique_pt_df, hour)
        print('examine unique %s pt at %sth hour' % (len(self.data), hour))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def read_df(self, args,df):
        data = {}
        cnt=0
        df = df.reset_index()

        for n, row in df.iterrows():
            stay = row['stay']
            if args.data_source in ['ms','apr_mimic3']:
                drg = self.d2i[row['DRG_CODE']]
                
                rw = self.d2w[row['DRG_CODE']] # relative weight
            else:
                if (not row['drg_code'] in self.d2i): 
                # print('not in')
                    cnt+=1
                    continue
                drg = self.d2i[row['drg_code']]
                rw = self.d2w[row['drg_code']] # relative weight
            

            age= row['AGE']
            if age>=300:
                age=age-300
            if args.data_source in ['ms','apr_mimic3']:
                los=row['LOS']
            else:
                los=row['los']
            

            X_text,lenX = self.read_text(args,stay)
            lenX=max(20,lenX)
            if args.LongFormer=='No':
                text=torch.tensor(X_text).long()
            else:
                text=X_text
            sample = {
                'lenX':torch.tensor(lenX).long(),
                'entry': stay,
                'text': text,
                'drg': torch.tensor(drg).long(),
                'age': torch.tensor(age).long(),
                'los': torch.tensor(los).long(),
                'rw': torch.tensor(rw).float()
            }


            data[n] = sample
        data={i:v for i,(_,v) in enumerate(data.items())}

        return data

    def read_text(self, args,stay):
        path = '%s/%s.dict' % (self.text_dir, stay)
        with open(path, 'rb') as f:
            text_dict = pk.load(f)

        
        tokens=' '.join(text_dict)
        if args.LongFormer=='Yes':
            return tokens,len(tokens)
        tokens = np.array(text_dict)
        X_text = np.zeros(self.max_seq_length)
        length = min(self.max_seq_length, len(tokens))
        X_text[:length] = tokens[:length]
        
        return X_text,length
        # return X_text

def load_rule(path):
    rule_df = pd.read_csv(path)
    
    # remove MDC 15 - neonate and couple other codes related to postcare
    if 'MS' in path:
        msk = (rule_df['MDC']!='15') & (~rule_df['MS-DRG'].isin([945, 946, 949, 950, 998, 999])) 
        space = sorted(rule_df[msk]['DRG_CODE'].unique())
    elif 'APR' in path:
        msk = (rule_df['MDC']!='15') & (~rule_df['APR-DRG'].isin([860, 863])) 
        space = sorted(rule_df[msk]['DRG_CODE'].unique())
        
    drg2idx = {}
    for d in space:
        drg2idx[d] = len(drg2idx)
    i2d = {v:k for k,v in drg2idx.items()}

    d2mdc, d2w = {}, {}
    for _, r in rule_df.iterrows():
        drg = r['DRG_CODE']
        mdc = r['MDC']
        w = r['WEIGHT']
        d2mdc[drg] = mdc
        d2w[drg] = w
        
    return rule_df, drg2idx, i2d, d2mdc, d2w

