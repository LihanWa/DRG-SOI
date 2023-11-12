import math
import torch 
import torch.nn as nn
import torch.nn.functional as Fu
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import sys
import numpy as np
from options import args


#****************************************************
class myModel(nn.Module):
    
    def __init__(self,args,embedding,Atten):
        super(myModel, self).__init__()
        self.data_source=args.data_source
        self.embed_size=200
        if self.data_source=='ms':
          self.age_embedding = nn.Embedding(91,8)  
          self.los_embedding = nn.Embedding(88,15)
        else:
          self.age_embedding = nn.Embedding(92, 8)  
          self.los_embedding = nn.Embedding(100, 15)   
        hidden_size0 = 256
        self.pblstm_encoder=nn.ModuleList([nn.LSTM(input_size = 4*hidden_size0, hidden_size=hidden_size0 ,num_layers = 2,bidirectional=True) for _ in range(3)])

        self.conv1 = nn.Conv1d(
            200, 512, kernel_size=5, stride=2, padding=1, bias=False
        )
        nn.init.xavier_uniform_(self.conv1.weight)

        self.bn0=torch.nn.BatchNorm1d(256)


        self.gelu = nn.GELU()
        self.gelu1 = nn.GELU()
        self.drop0=nn.Dropout(0.05)
        self.drop1=nn.Dropout(0.1)
        self.drop2=nn.Dropout(0.15)
        self.drop3=nn.Dropout(0.2)
        self.drop4=nn.Dropout(0.25)
        self.drop_pl=nn.Dropout(0)
        self.drop_emb=nn.Dropout(0)
        self.drop_c1=nn.Dropout(0.1)
        self.dropc12=nn.Dropout(0.2)
        self.drop_los=nn.Dropout(0.2)
        self.drop_age=nn.Dropout(0.15)
        self.drop_age2=nn.Dropout(0.25)
        self.drop_at=nn.Dropout(0.15)
        self.drop_at1=nn.Dropout(0.2)
        self.drop_at2=nn.Dropout(0.25)
        self.drop_ln0=nn.Dropout(0.15)
        self.drop_ln01=nn.Dropout(0.2)
        self.drop_ln02=nn.Dropout(0.25)
        self.drop_ln1=nn.Dropout(0.2)
        self.drop_ln12=nn.Dropout(0.3)
        self.cnt_f=0
        self.embed_num, self.embed_size = embedding.shape
        weight = torch.from_numpy(embedding).float()
        print(self.embed_num)
        self.embed = nn.Embedding(embedding_dim=self.embed_size, num_embeddings=self.embed_num, padding_idx=0, _weight=weight)
        self.embed.weight.requires_grad=True
        self.relu=nn.ReLU()
        self.attend=Atten
        self.layer_norm0 = nn.LayerNorm(normalized_shape=512)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=512)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=662)
        if self.data_source=='ms':
          self.layer_norm3 = nn.LayerNorm(normalized_shape=738)
          self.layer_norm4 = nn.LayerNorm(normalized_shape=738)
          self.linear=nn.Linear(hidden_size0*2+150+23,738)
          self.linear2=nn.Linear(738,738)
        else:
          self.layer_norm3 = nn.LayerNorm(normalized_shape=1136)
          self.layer_norm4 = nn.LayerNorm(normalized_shape=1136)
          self.linear=nn.Linear(hidden_size0*2+150+23,1136)
          self.linear2=nn.Linear(1136,1136)
        self.flat=nn.Flatten()

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        
        self.pool=CustomAveragePooling()
    def max_pooling_downsample(self,tensor, lx):
          B, T, N = tensor.size()
          target_length=T//2
          real_lengths = lx
          processed_list = []
          for i in range(B):
            real_length=real_lengths[i]
            tmp=real_length
            if real_lengths[i] > target_length:
              real_length=real_lengths[i]
              tmp=real_length
              to_down_l=(real_length-target_length)*2
              toDown_tensor = tensor[i, :to_down_l, :]
              kernel_size = 2
              stride = 2
              unfolded = toDown_tensor.unfold(0, kernel_size, stride)
              unfolded=unfolded.transpose(1,2)
              pooled, _ = unfolded.max(1)
              left = tensor[i, to_down_l:real_length, :] 
              pooled = torch.cat([pooled, left], dim=0)
              processed_list.append(pooled)
              lx[i]=target_length
            else:

              processed_list.append(tensor[i, :target_length, :])
           
          return torch.stack(processed_list),lx

    def forward(self,criterion1,x,lx, age,los,target,cnt):
        x=x.cuda()
        age=(age).cuda()
        los=los.cuda()
        target=target.cuda()
        

        self.cnt_f=cnt
        # print(x.shape)
        x=self.embed(x)
        # print(x)
        age_embed=self.age_embedding(age).squeeze(1)
        
        if self.cnt_f==1 or self.cnt_f==2: self.drop_age=self.drop_age2
        age_embed=self.drop_age(age_embed)

        los_embed=self.los_embedding(los).squeeze(1)
        los_embed=self.drop_los(los_embed)
        if self.cnt_f==0: self.drop_emb=self.drop0
        if self.cnt_f==1: self.drop_emb=self.drop1
        if self.cnt_f==2: self.drop_emb=self.drop2
        x=self.drop_emb(x)

        x=x.transpose(1,2)
        conv1=self.conv1(x)
        out=conv1.transpose(1,2)
        out=self.layer_norm0(out)
        out=self.gelu(out)
        if self.cnt_f==1 or self.cnt_f==2: self.drop_c1=self.dropc12
        out=self.drop_c1(out)


        lx=(lx//2-3)
        maxD,lx=self.max_pooling_downsample(out,lx)
        output=maxD
        for i in range(2):
          if(self.training):
            output = output.clone()
            mask = output.new_empty(output.size(0), 1, output.size(2), requires_grad=False).bernoulli_(1 - 0.25)
            mask = mask.div_(1 - 0.25)
            mask = mask.expand_as(output)
            output=output*mask

          lx=lx//2
          B,T,F=output.shape
          T=T//2*2
          output=output[:,:T,:]
          output=output.reshape((B,T//2,F*2))

          packed = pack_padded_sequence(output, lx, batch_first=True, enforce_sorted=False)
          output, (h_n, c_n) = self.pblstm_encoder[i](packed)  

          output, lx = pad_packed_sequence(output, batch_first=True)


        out_hid=output
        attn_context, attn_weights = self.attend.compute_context(out_hid) 
        torch.save(attn_weights, 'tensor.pt')
        att_hid= torch.cat((attn_context,out_hid),dim=2)
        lyar=self.layer_norm2(att_hid)
        if self.cnt_f==1: self.drop_at=self.drop_at1
        if self.cnt_f==2: self.drop_at=self.drop_at2
        lyar=self.drop_at(lyar)
        pooled_output = self.pool(lyar, lx)

        sq=pooled_output.squeeze(1)
        if self.cnt_f==0: self.drop_pl=self.drop1
        if self.cnt_f==1: self.drop_pl=self.drop2
        if self.cnt_f==2: self.drop_pl=self.drop3
        if self.cnt_f==0: self.drop_ln0=self.drop2
        if self.cnt_f==1: self.drop_ln0=self.drop2
        if self.cnt_f==2: self.drop_ln0=self.drop4
        sq=self.drop_pl(sq)
        combined = torch.cat([sq,los_embed,age_embed], dim=1)
        
        ln= self.linear(combined)
        ln=self.layer_norm3(ln)
        ln=self.gelu(ln)
        if self.cnt_f==1: self.drop_ln0=self.drop_ln01
        if self.cnt_f==2: self.drop_ln0=self.drop_ln02
        ln=self.drop_ln0(ln)
        
        ln2=self.linear2(ln)
        ln2=self.layer_norm4(ln2)
        ln2=self.gelu(ln2)
        if self.cnt_f==1 or self.cnt_f==2: self.drop_ln1=self.drop_ln12
        ln2=self.drop_ln1(ln2)

        loss1=0.3*criterion1(ln,target)
        logit=ln2
        loss = Fu.cross_entropy(logit, target)

        return logit, loss,loss1,self.cnt_f

class Attention(nn.Module):
  def __init__(self,input_size, output_size):
    super().__init__()
    self.KW = nn.Linear(input_size,output_size)
    self.QW = nn.Linear(input_size,output_size)
    self.VW = nn.Linear(input_size,output_size)


    
  def compute_context(self, out_hid):
    self.query=self.QW(out_hid)
    self.key=self.KW(out_hid)
    self.value=self.VW(out_hid)
    qk=torch.bmm(self.query,self.key.transpose(1,2)) #B*1*nT

    attn_weights=torch.nn.functional.softmax(qk/qk.shape[2]**0.5,dim=2)
    attn_context=torch.bmm(attn_weights,self.value)
    attn_weights=attn_weights[1]
    return attn_context,attn_weights
#########


def plot_attention(attention):
    plt.clf()
    sns.heatmap(attention.detach().cpu().numpy(), cmap='GnBu')

    plt.show()

class CustomAveragePooling(torch.nn.Module):
    def __init__(self):
        super(CustomAveragePooling, self).__init__()

    def forward(self, x, lengths):
        batch_size, seq_count, seq_len = x.size()
        mask = torch.arange(seq_count).unsqueeze(0).expand(batch_size, seq_count).float() < lengths.unsqueeze(1).float()
        mask=mask.cuda()
        lengths=lengths.cuda()
        valid_sum = (x * mask.unsqueeze(2).float()).sum(dim=1)
        pooled = valid_sum / lengths.unsqueeze(1).float()

        return pooled


import torch.nn.functional as F


def model(args, embedding,Atten,num):
    # if num==2:
    #   model=myModel2(args,embedding,Atten)
    # else:
    model=myModel(args,embedding,Atten)
    return model