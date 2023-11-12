import pandas as pd
import numpy as np
import pickle as pk
import os
import time
import math
from tqdm.auto import tqdm
import copy 
from options import args

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,TensorDataset
from argparse import Namespace


from eval_utils import full_metrics, result2str, reg_metrics
from data_utils_ori import load_rule
class CenterLoss(nn.Module):
    """Center Loss
        Center Loss Paper:
        https://ydwen.github.io/papers/WenECCV16.pdf
    Args:
        nn (_type_): _description_
    """
    def __init__(self, 
                 num_classes=1136, 
                 feat_dim=1136, 
                 ) -> None:
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        
        centers = self.centers[labels]
        dist = torch.pow((x.cuda() - centers.cuda()),2).sum(dim=1,keepdim=True).sqrt()
        dist = torch.clamp(dist, min=1e-12, max=1e+12)
        
        loss = dist.sum()/(2*len(dist))
        return loss
def eval_test(model, dataset, score_func, rule_path=None, print_results=False, batch_size=16):
    loss, score, (y_pred, y, stays,lxs) = test(model,dataset, batch_size, score_func)

    inf = {'y_pred':y_pred,'y':y,'stay':stays}

    if rule_path == None:
        return score, rule_path
    else:

        rule_df, d2i, i2d, d2mdc, d2w = load_rule(rule_path)
        result_dict = full_metrics(y_pred, y, rule_df, d2i)



    if print_results:
        print(result2str(result_dict))

    return result_dict, inf  
def softmax(x):
    e_x = np.exp(x)
    return e_x / np.expand_dims(e_x.sum(axis=1), 1)
def train_dev(model, train_dataset, dev_dataset, epochs, batch_size, optimizer, score_func):


    best_score=0
    best_loss=1e+5
    better=True
    cnt_w=2
    if args.data_source=='ms':
        criterion1 = CenterLoss(num_classes=738, feat_dim=738)
    else:
        criterion1 = CenterLoss(num_classes=1136, feat_dim=1136)

    optimizer1 = torch.optim.Adam(criterion1.parameters(), lr = 1e-4)
    for epoch in range(args.epochs):
        print("training on epoch", epoch+1)
        since = time.time()

        tr_loss = train(model, train_dataset, batch_size,criterion1, optimizer,optimizer1,cnt_w)
        dev_loss, dev_score, _ = test(model, dev_dataset, batch_size, score_func)
        if dev_score>best_score or dev_loss<best_loss:
     
          better=True
          best_score=dev_score
          best_loss=dev_loss
          if args.LongFormer=='Yes':
              save_file='best%s_LongFormer' %args.data_source.upper()                       
          else:
              save_file='best%s' %args.data_source.upper()                       
          torch.save({'model_state_dict':model.state_dict()},save_file)
        else:
          cnt_w+=1
          if cnt_w==3:
            better=False
            cnt_w=2
        if better==False:
          current_lr = optimizer.param_groups[0]['lr']*0.9
          for param_group in optimizer.param_groups:
              param_group['lr'] = current_lr
          current_lr = optimizer1.param_groups[0]['lr']*0.9
          for param_group in optimizer1.param_groups:
              param_group['lr'] = current_lr
          better=True

        better = 'No..'
        cond1 = dev_score > best_score
        cond2 = dev_score < best_score
        if cond1 or cond2:
            best_score = dev_score
            better = 'Yes!'
            count = epoch+1

        time_elapsed = time.time() - since
        to_print = (time_elapsed // 60, time_elapsed % 60, tr_loss, dev_loss, dev_score, cnt_w)
        print("finish in {:.0f}m{:.0f}s tr_loss: {:.3f}, dev_loss: {:.3f}, dev_score: {:.3f}...better? -> {}".format(*to_print))
        print(optimizer.param_groups[0]['lr'])
        print(optimizer1.param_groups[0]['lr'])



def train(model, dataset, batch_size,criterion1, optimizer,optimizer1,cnt):
    tr_loss, nb_tr_steps = 0., 0.

    model=model.cuda()
    model.train()
    

    dloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True,pin_memory  = True,num_workers=8,drop_last=True)
    batch_bar   = tqdm(total=len(dloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    for i,batch in enumerate(dloader):
        if args.LongFormer=='No':
            x = batch['text']
            lx=batch['lenX']
            age=batch['age']
            los=batch['los']
            scaler = torch.cuda.amp.GradScaler() 
            with torch.cuda.amp.autocast():
                label = batch['drg']

            optimizer.zero_grad()
            optimizer1.zero_grad()
            logits, loss,loss1,cnt_w = model(criterion1,x ,lx,age,los,label,cnt)
            loss_sum=loss+loss1
            scaler.scale(loss_sum).backward()
            scaler.step(optimizer1)
            del x, lx, label
        else:
            if i%700==0 and i>0:
              print('save')
              save_file='best%s_LongFormer' %args.data_source.upper()                      
              torch.save({'model_state_dict':model.state_dict()},save_file)
       
            scaler = torch.cuda.amp.GradScaler() 
            batch['text']['input_ids']=batch['text']['input_ids'].cuda()
            batch['text']['attention_mask']=batch['text']['attention_mask'].cuda()

            x = batch['text']

            labels = batch['drg'].cuda()

            optimizer.zero_grad()

           
            tmp = model(**x, labels=labels)
            loss=tmp.loss
            logits=tmp.logits
            scaler.scale(loss).backward()
            del batch, labels, logits,tmp
        
        scaler.step(optimizer)
        
        scaler.update()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)


        tr_loss += loss.item()
        nb_tr_steps += 1

        batch_bar.set_postfix(loss="{:.04f}".format(float(tr_loss / (nb_tr_steps + 1))),
        lr="{:.05f}".format(float(optimizer.param_groups[0]['lr'])))
        batch_bar.update()
        torch.cuda.empty_cache()

    batch_bar.close()

    tr_loss /= nb_tr_steps
    return tr_loss 



        
        




def test(model, dataset, batch_size, score_func):
    te_loss, nb_te_steps = 0., 0.
    y_pred, y, stays ,lxs= [], [], [],[]
    model.cuda()
    model.eval()

    dloader =  DataLoader(dataset, batch_size=batch_size, shuffle=False,pin_memory  = True,num_workers=8,drop_last=True)
    for batch in dloader:
        
        if args.LongFormer=='No':
            x = batch['text']
            lx=batch['lenX']
            label = batch['drg']
            entry  = batch['entry']

            age=batch['age']
            los=batch['los']
            if args.data_source=='ms':
                criterion1 = CenterLoss(num_classes=738, feat_dim=738)
            else:
                criterion1 = CenterLoss(num_classes=1136, feat_dim=1136)

            with torch.no_grad():
                logit, loss,loss1,cnt_w = model(criterion1,x, lx,age,los,label,0)
        else:
            batch['text']['input_ids']=batch['text']['input_ids'].cuda()
            batch['text']['attention_mask']=batch['text']['attention_mask'].cuda()
            x = batch['text']
            label = batch['drg'].cuda()
            entry = batch['entry']

            with torch.no_grad():
                tmp = model(**x, labels=label)

            loss=tmp.loss
            logit=tmp.logits

        te_loss += loss.item()
        nb_te_steps += 1

        logit = logit.detach().cpu().numpy()
        

        label = label.cpu().numpy()
        y_pred.append(logit)
        y.append(label)
        stays.append(entry)

    y_pred = np.concatenate(y_pred)
    y = np.concatenate(y)
    stays = np.concatenate(stays)

 
    te_loss /= nb_te_steps

    te_score = score_func(y_pred, y)
    torch.cuda.empty_cache()
    return te_loss, te_score, (y_pred, y, stays,lxs)


      
        



def split_df_by_pt_mimic3(df, frac=None, k=None):
    pt_count = df.groupby(['SUBJECT_ID']).size()
    pt_multi = pd.Series(pt_count[pt_count >1].index)
    pt_single= pd.Series(pt_count[pt_count==1].index)

    
    assert len(pt_single) + len(pt_multi) == len(df.SUBJECT_ID.unique())

    if frac:
        test_multi = pt_multi.sample(frac=frac, random_state=1443)
        test_single = pt_single.sample(frac=frac, random_state=1443)

        test_pt = test_single.append(test_multi)
        test_mask = df.SUBJECT_ID.isin(test_pt)

        test_df = df[test_mask]
        train_df = df[~test_mask]

        return train_df, test_df

    elif k:
        np.random.RandomState(seed=1443).shuffle(pt_multi)
        np.random.RandomState(seed=1443).shuffle(pt_single)

        tick1 = int( len(pt_multi) / k )
        tick2 = int( len(pt_single)/ k )

        pt_multi10 = int( len(pt_multi) * .1)
        pt_single10= int( len(pt_single)* .1)

        splits = []    
        i = 0
        while i < k-1:
            subj1 = pt_multi[i*tick1 : (i+1)*tick1]
            subj2 = pt_single[i*tick2 : (i+1)*tick2]
            test_subj = pd.concat([subj1, subj2])

            train_df = df[~df.SUBJECT_ID.isin(test_subj)]
            test_df = df[df.SUBJECT_ID.isin(test_subj)]

            test_subj10=pd.concat([subj1[:pt_multi10], subj2[:pt_single10]])
            train_df90 = df[~df.SUBJECT_ID.isin(test_subj10)]
            test_df10 = df[df.SUBJECT_ID.isin(test_subj10)]

            splits.append((train_df, test_df, train_df90, test_df10))
            i+=1

        subj1 = pt_multi[i*tick1 : ]
        subj2 = pt_single[i*tick2 : ]
        test_subj = pd.concat([subj1, subj2])

        train_df = df[~df.SUBJECT_ID.isin(test_subj)]
        test_df = df[df.SUBJECT_ID.isin(test_subj)]

        test_subj10=pd.concat([subj1[:pt_multi10], subj2[:pt_single10]])
        train_df90 = df[~df.SUBJECT_ID.isin(test_subj10)]
        test_df10 = df[df.SUBJECT_ID.isin(test_subj10)]

        splits.append((train_df, test_df, train_df90, test_df10))
        
        assert len(splits) == k
        return splits

def split_df_by_pt_mimic4(df, frac=None, k=None):
    pt_count = df.groupby(['subject_id']).size()
    pt_multi = pd.Series(pt_count[pt_count >1].index)
    pt_single= pd.Series(pt_count[pt_count==1].index)

    
    assert len(pt_single) + len(pt_multi) == len(df.subject_id.unique())

    if frac:
        test_multi = pt_multi.sample(frac=frac, random_state=1443)
        test_single = pt_single.sample(frac=frac, random_state=1443)

        test_pt = test_single.append(test_multi)
        test_mask = df.subject_id.isin(test_pt)

        test_df = df[test_mask]
        train_df = df[~test_mask]

        return train_df, test_df

    elif k:
        np.random.RandomState(seed=1443).shuffle(pt_multi)
        np.random.RandomState(seed=1443).shuffle(pt_single)

        tick1 = int( len(pt_multi) / k )
        tick2 = int( len(pt_single)/ k )

        pt_multi10 = int( len(pt_multi) * .1)
        pt_single10= int( len(pt_single)* .1)

        splits = []    
        i = 0
        while i < k-1:
            subj1 = pt_multi[i*tick1 : (i+1)*tick1]
            subj2 = pt_single[i*tick2 : (i+1)*tick2]
            test_subj = pd.concat([subj1, subj2])

            train_df = df[~df.subject_id.isin(test_subj)]
            test_df = df[df.subject_id.isin(test_subj)]

            test_subj10=pd.concat([subj1[:pt_multi10], subj2[:pt_single10]])
            train_df90 = df[~df.subject_id.isin(test_subj10)]
            test_df10 = df[df.subject_id.isin(test_subj10)]

            splits.append((train_df, test_df, train_df90, test_df10))
            i+=1

        subj1 = pt_multi[i*tick1 : ]
        subj2 = pt_single[i*tick2 : ]
        test_subj = pd.concat([subj1, subj2])

        train_df = df[~df.subject_id.isin(test_subj)]
        test_df = df[df.subject_id.isin(test_subj)]

        test_subj10=pd.concat([subj1[:pt_multi10], subj2[:pt_single10]])
        train_df90 = df[~df.subject_id.isin(test_subj10)]
        test_df10 = df[df.subject_id.isin(test_subj10)]

        splits.append((train_df, test_df, train_df90, test_df10))
        
        assert len(splits) == k
        return splits



