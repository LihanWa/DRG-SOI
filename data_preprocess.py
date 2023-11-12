import pandas as pd 
import numpy as np
import os 
import pickle as pk 
import random
import argparse
from tqdm import tqdm

from options import args


import re
import nltk
import ipdb
import Levenshtein

from nltk.stem import WordNetLemmatizer as wnl
from nltk.stem import PorterStemmer as ps
from collections import Counter, OrderedDict
from nltk import sent_tokenize, word_tokenize
from gensim.models import KeyedVectors

def main():

    if args.collectText=='Yes':    
        print('Collect texts for %s drg' % args.data_source)
        df = getDF(args.data_source, args.mimic_dir)
        if args.data_source in ['ms','apr_mimic3']:
            notes_df = loadNOTES_mimic3(df[['SUBJECT_ID', 'HADM_ID', 'INTIME']])
        else:
            notes_df = loadNOTES_mimic4(df[['subject_id', 'hadm_id', 'intime']])
        if args.data_source in ['ms','apr_mimic3']:
            collectText_mimic3(df, notes_df, args.data_source)
        else:
            collectText_mimic4(df, notes_df, args.data_source)

    print('\n\n\n')
    if args.LongFormer=='No':
        data_dir = '%s/%s' % ('data', args.data_source)
        text_dir = '%s/text_raw' % data_dir
        output_dir = '%s/text_tokens' % data_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        files = [f for f in os.listdir(text_dir) if f.endswith('pk')]
        words = get_common(files, text_dir, output_dir)
        token2id = get_embeddings(words, output_dir)
        for file in tqdm(files):
            save2id(file, token2id, text_dir, output_dir)
    else:
        data_dir = '%s/%s' % ('data', args.data_source)
        text_dir = '%s/text_raw' % data_dir
        output_dir = '%s/text_LongFormer' % data_dir
        filename = "abbreviation-fullspelling.txt"  
        abbreviation_dict={}
        if args.use_AbbFull=='Yes':
            abbreviation_dict = load_abbreviation_dict_from_file(filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        files = [f for f in os.listdir(text_dir) if f.endswith('pk')]
        for file in tqdm(files):
            LongFomrer_save2id(file, text_dir, output_dir,abbreviation_dict)

def replace_tokens_with_expansion(tokens, abbreviation_dict):
    expanded_tokens = []

    for token in tokens:
        if token in abbreviation_dict:
            expanded_tokens.extend(abbreviation_dict[token])  
        else:
            expanded_tokens.append(token)

    return expanded_tokens
def load_abbreviation_dict_from_file(filename):
    abbreviation_dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            head, _, tail = line.rpartition(':')
            if head and tail:
                abbreviation_dict[head.strip()] =  tail.strip().lower().split()
    return abbreviation_dict

SECTION_TITLES = re.compile(
    r'('
    r'ABDOMEN AND PELVIS|CLINICAL HISTORY|CLINICAL INDICATION|COMPARISON|COMPARISON STUDY DATE'
    r'|EXAM|EXAMINATION|FINDINGS|HISTORY|IMPRESSION|INDICATION'
    r'|MEDICAL CONDITION|PROCEDURE|REASON FOR EXAM|REASON FOR STUDY|REASON FOR THIS EXAMINATION'
    r'|TECHNIQUE'
    r'):|FINAL REPORT',
    re.I | re.M)


def pattern_repl(matchobj):
    """
    Return a replacement string to be used for match object
    """
    return ' '.rjust(len(matchobj.group(0)))


def find_end(text):
    """Find the end of the report."""
    ends = [len(text)]
    patterns = [
        re.compile(r'BY ELECTRONICALLY SIGNING THIS REPORT', re.I),
        re.compile(r'\n {3,}DR.', re.I),
        re.compile(r'[ ]{1,}RADLINE ', re.I),
        re.compile(r'.*electronically signed on', re.I),
        re.compile(r'M\[0KM\[0KM')
    ]
    for pattern in patterns:
        matchobj = pattern.search(text)
        if matchobj:
            ends.append(matchobj.start())
    return min(ends)


def split_heading(text):
    """Split the report into sections"""
    start = 0
    for matcher in SECTION_TITLES.finditer(text):
        # add last
        end = matcher.start()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        # add title
        start = end
        end = matcher.end()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        start = end

    # add last piece
    end = len(text)
    if start < end:
        section = text[start:end].strip()
        if section:
            yield section


def clean_text(text):
    text = re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)
    text = re.sub(r'_', ' ', text)
    pattern = r'\s{6,}'  
    text = re.sub(pattern, ' / ', text)
    start = 0
    end = find_end(text)
    new_text = ''
    if start > 0:
        new_text += ' ' * start
    new_text = text[start:end]

    if len(text) - end > 0:
        new_text += ' ' * (len(text) - end)
    return new_text

def preprocess_mimic(text):
    """
    Preprocess reports in MIMIC-III.
    1. remove [**Patterns**] and signature
    2. split the report into sections
    3. tokenize sentences and words
    4. lowercase
    """
    tokens = []
    for sec in split_heading(clean_text(text)):
        for sent in sent_tokenize(sec):
            tokens.extend([w.lower() for w in word_tokenize(sent)])
    # lemmatizer=wnl()

    # # # stemmer=ps()
    # # # stems=[stemmer.stem(word=word) for word in words]
    # lemmatized=[lemmatizer.lemmatize(word=word,pos='v') for word in tokens]
    return tokens
# ====================



def get_mimic3_stay_tokens(file, text_dir,abbreviation_dict=None):
    """
        input: path
        output: tokens in order for all texts representing the stay
    """
    content = pd.read_pickle(os.path.join(text_dir, file))
    
    similarity_threshold=0.35
   
    ori=len(content['TEXT'])
    now=0
    left=ori-now
    
    l=0

    while(l<2000 and left>0):
        unique_sentences = []
        for i in range(len(content['TEXT'])):
            sentence=content['TEXT'][i]
            category=content['CATEGORY'][i]
            is_duplicate = False
            for unique_sentence in unique_sentences:
                similarity = Levenshtein.ratio(sentence, unique_sentence)
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:

                unique_sentences.append(category+" : "+sentence)
            if i==len(content['TEXT'])-1:
                res=[]
                l=0

                for j in unique_sentences:
                    j=preprocess_mimic(j)
                    res.extend(j)
                    l+=(len(j))
                now=len(unique_sentences)
                similarity_threshold+=0.1
                left=ori-now
    if args.LongFormer=='Yes':
        res=replace_tokens_with_expansion(res,abbreviation_dict)
    return res
def get_mimic4_stay_tokens(file, text_dir,abbreviation_dict=None):
    """
        input: path
        output: tokens in order for all texts representing the stay
    """
    content = pd.read_pickle(os.path.join(text_dir, file))
    
    similarity_threshold=0.35

    ori=len(content['text'])
    now=0
    left=ori-now
    
    l=0
    while(l<2000 and left>0):
        unique_sentences = []
        for i in range(len(content['text'])):
            sentence=content['text'].iloc[i]
            # category=content['CATEGORY'][i]
            # print(category)
            is_duplicate = False
            for unique_sentence in unique_sentences:
                similarity = Levenshtein.ratio(sentence, unique_sentence)
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:

                unique_sentences.append(sentence)
            if i==len(content['text'])-1:
                res=[]
                l=0
                for j in unique_sentences:
                    j=preprocess_mimic(j)
                    res.extend(j)
                    l+=(len(j))
                now=len(unique_sentences)
                similarity_threshold+=0.1
                left=ori-now
    if args.LongFormer=='Yes':
        res=replace_tokens_with_expansion(res,abbreviation_dict)
    return res


def get_common(files, text_dir, output_dir):
    all_tokens=[]
    for file in tqdm(files):
        if args.data_source in ['ms','apr_mimic3']:
            all_tokens.extend(get_mimic3_stay_tokens(file, text_dir))
        else:
            all_tokens.extend(get_mimic4_stay_tokens(file, text_dir))
        
    token_count = Counter(all_tokens)

    common = [w for (w,c) in token_count.most_common() if c >= 3]  

    print("{} tokens in text, {} unique, and {} of them appeared at least three times".format(len(all_tokens), len(token_count),len(common)))
    with open(os.path.join(output_dir, 'unique_common.txt'), 'w') as f:
        for w in common:
            f.write(w+'\n')
    return common

def get_embeddings(words, output_dir):
    print("loading biovec...")
    model = KeyedVectors.load_word2vec_format(os.path.join('..', 'BioWordVec_PubMed_MIMICIII_d200.vec.bin'), binary=True)
    print("loaded, start to get embed for tokens")

    model_vocab = set(model.index_to_key)

    valid_words = []
    oov = []
    for w in words:
        if w in model_vocab:
            valid_words.append(w)
        else:
            oov.append(w)
    print("oov", oov)

    token2id = {}
    token2id['<pad>'] = 0
    for word in valid_words:
        token2id[word] = len(token2id)
    token2id['<unk>'] = len(token2id)

    dim = model.vectors.shape[1]
    embedding = np.zeros( (len(valid_words)+2, dim), dtype=np.float32)
    embedding[0] = np.zeros(dim,)
    embedding[-1] = np.random.randn(dim,)
    print("embed shape", embedding.shape)
    for i, w in enumerate(valid_words):
        embedding[i+1] = model[w]

    t2i_path = os.path.join(output_dir, 'token2id2800.dict') 
    with open(t2i_path, 'wb') as f:
        pk.dump(token2id, f)

    embed_path = os.path.join(output_dir, 'embeddingAug.npy')
    np.save(embed_path, embedding)

    return token2id

def save2id(file, token2id, text_dir, output_dir):
    output_path = os.path.join(output_dir, file.replace('pk','dict'))
    
    if args.data_source in ['ms','apr_mimic3']:
        tokens = get_mimic3_stay_tokens(file, text_dir)
    else:
        tokens = get_mimic4_stay_tokens(file, text_dir)
    output=[token2id[w] if w in token2id else token2id['<unk>'] for w in tokens]
    with open(output_path, 'wb') as f:
        pk.dump(output, f)
def LongFomrer_save2id(file, text_dir, output_dir,abbreviation_dict):
    output_path = os.path.join(output_dir, file.replace('pk','dict'))
    if args.data_source in ['ms','apr_mimic3']:
        tokens = get_mimic3_stay_tokens(file, text_dir,abbreviation_dict)
    else:
        tokens = get_mimic4_stay_tokens(file, text_dir,abbreviation_dict)
    output=tokens
    with open(output_path, 'wb') as f:
        pk.dump(output, f)
def collectText_mimic3(drg_df, notes_df, drg_type='ms'):
    drg_path = '%s/%s' % (args.data_dir, drg_type)
    text_dir = '%s/text_raw/' % drg_path
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)

    pairs = []
    for _, row in tqdm(drg_df.iterrows(), total=len(drg_df)):
        sub = row['SUBJECT_ID']
        hadm = row['HADM_ID']
        stay = row['stay']
        adm_diag = row['DIAGNOSIS']

        hours = extract_note_append_adm_diag_mimic3(sub, hadm, adm_diag, notes_df, text_dir)

        if hours:
            pairs.append((stay, *hours))

    df_tmp = pd.DataFrame.from_records(pairs, columns=['stay','hour0','hour12','hour24','hour36'])
    drg_df_h = pd.merge(drg_df, df_tmp, on=['stay'])
    print("at least one note before the 48h threshold at icu")

    pt, st = len(drg_df_h.SUBJECT_ID.unique()), len(drg_df_h)
    print("..there are {} pt, {} stays w/ {} drg in total".format(pt, st, drg_type))
    split_cohort_mimic3(drg_df_h, drg_path) 

def collectText_mimic4(drg_df, notes_df, drg_type='apr_mimic4'):
    drg_path = '%s/%s' % (args.data_dir, drg_type)
    text_dir = '%s/text_raw/' % drg_path
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)

    pairs = []
    for _, row in tqdm(drg_df.iterrows(), total=len(drg_df)):
        sub = row['subject_id']
        hadm = row['hadm_id']
        stay = row['stay']


        hours = extract_note_append_adm_diag_mimic4(sub, hadm, notes_df, text_dir)

        if hours:
            pairs.append((stay, *hours))

    df_tmp = pd.DataFrame.from_records(pairs, columns=['stay','hour0','hour12','hour24','hour36'])
    drg_df_h = pd.merge(drg_df, df_tmp, on=['stay'])
    print("at least one note before the 48h threshold at icu")

    pt, st = len(drg_df_h.subject_id.unique()), len(drg_df_h)
    print("..there are {} pt, {} stays w/ {} drg in total".format(pt, st, drg_type))

    split_cohort_mimic4(drg_df_h, drg_path) 

def getDF(drg_type, mimic_dir):
    if drg_type == 'ms':
        drg_df = getMS(mimic_dir)
    elif drg_type == 'apr_mimic3':
        drg_df = getAPR_mimic3(mimic_dir)
    elif drg_type == 'apr_mimic4':
        drg_df = getAPR_mimic4(mimic_dir)
    if args.data_source in ['ms','apr_mimic3']:
        return filterCohort_mimic3(drg_df, mimic_dir)
    else:
        return filterCohort_mimic4(drg_df, mimic_dir)

def getMS(mimic_dir):
    drg = pd.read_csv(os.path.join(mimic_dir, 'DRGCODES.csv'), usecols=['SUBJECT_ID', 'HADM_ID','DRG_TYPE', 'DRG_CODE', 'DESCRIPTION'])
    ms_df = drg[drg.DRG_TYPE == 'MS'] 
    ms_df = ms_df.dropna() 
    drg_df = ms_df[['SUBJECT_ID', 'HADM_ID','DRG_CODE']] 
    print("raw cases w/ MS drg: pt - {}, stays - {}".format( len(drg_df.SUBJECT_ID.unique()), len(drg_df)))

    return drg_df

def getAPR_mimic3(mimic_dir):
    drg = pd.read_csv(os.path.join(mimic_dir, 'DRGCODES.csv'), usecols=['SUBJECT_ID', 'HADM_ID','DRG_TYPE', 'DRG_CODE', 'DESCRIPTION'])
    apr_df = drg[drg.DRG_TYPE == 'APR '] 
    apr_df = apr_df.drop_duplicates(subset=['HADM_ID','DRG_CODE', 'DESCRIPTION'])
    raw = len(apr_df)


    apr_df = apr_df.sort_values(by=['HADM_ID', 'DESCRIPTION'], ascending=False).drop_duplicates(subset=['HADM_ID', 'DESCRIPTION'])
    print("{} raw apr codes in MIMIC, {} after first dropping duplicated severity".format(raw, len(apr_df)))
 
    dup_mask = apr_df[apr_df.duplicated(subset=['HADM_ID'])].HADM_ID
    apr_df = apr_df[~apr_df.HADM_ID.isin(dup_mask)]
    drg_df = apr_df[['SUBJECT_ID', 'HADM_ID','DRG_CODE']]
    print("raw cases w/ APR drg: pt - {}, stays - {}".format( len(drg_df.SUBJECT_ID.unique()), len(drg_df)))

    return drg_df
def getAPR_mimic4(mimic_dir):
    drg = pd.read_csv(os.path.join(mimic_dir, 'DRGCODES.csv'), usecols=['subject_id', 'hadm_id','drg_type', 'drg_code', 'description'])
    apr_df = drg[drg.drg_type == 'APR'] 
    apr_df = apr_df.drop_duplicates(subset=['hadm_id','drg_code', 'description'])
    raw = len(apr_df)
    
    apr_df = apr_df.sort_values(by=['hadm_id', 'description'], ascending=False).drop_duplicates(subset=['hadm_id', 'description'])
    print("{} raw apr codes in MIMIC, {} after first dropping duplicated severity".format(raw, len(apr_df)))

    dup_mask = apr_df[apr_df.duplicated(subset=['hadm_id'])].hadm_id
    apr_df = apr_df[~apr_df.hadm_id.isin(dup_mask)]
    drg_df = apr_df[['subject_id', 'hadm_id','drg_code']]
    print("raw cases w/ APR drg: pt - {}, stays - {}".format( len(drg_df.subject_id.unique()), len(drg_df)))

    return drg_df

def filterCohort_mimic3(drg_df, mimic_dir):
    """
        1. filter adult
        2. filter single ICU stay - only one icu stay & no transfer

        return: df of stay information
    """

    icu = pd.read_csv(os.path.join(mimic_dir, 'ICUSTAYS.csv'))
    icu_count = icu.groupby(['HADM_ID']).size()
    icu_once = icu.HADM_ID.isin(icu_count[icu_count==1].index)
    icu_no_transfer = (icu.FIRST_WARDID == icu.LAST_WARDID) & (icu.FIRST_CAREUNIT == icu.LAST_CAREUNIT)
    icu_single = icu[icu_once & icu_no_transfer][['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID','INTIME', 'DBSOURCE','LOS']]
    adm = pd.read_csv(os.path.join(mimic_dir, 'ADMISSIONS.csv'), usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME','DISCHTIME','DIAGNOSIS'])
    pt = pd.read_csv(os.path.join(mimic_dir, 'PATIENTS.csv'), usecols=['SUBJECT_ID','DOB'])

    drg_adm = pd.merge(drg_df, adm, on=['SUBJECT_ID', 'HADM_ID'])
    drg_dob = pd.merge(drg_adm, pt, on=['SUBJECT_ID'])
    admtime = pd.to_datetime(drg_dob.ADMITTIME).dt.date.apply(np.datetime64)
    dobtime = pd.to_datetime(drg_dob.DOB).dt.date.apply(np.datetime64)

    drg_dob['ADMITTIME'] = pd.to_datetime(drg_dob['ADMITTIME']).dt.date
    drg_dob['DOB'] = pd.to_datetime(drg_dob['DOB']).dt.date
    drg_dob['AGE']  = drg_dob.apply(lambda e: (e['ADMITTIME'] - e['DOB']).days/365, axis=1)


    drg_adult = drg_dob[drg_dob.AGE >= 18]
    print("age over 18 w/ drg: pt - {}, visits - {}".format(len(drg_adult.SUBJECT_ID.unique()), len(drg_adult) ))

    drg_final = pd.merge(drg_adult, icu_single, on=['SUBJECT_ID', 'HADM_ID'])
    print("plus criterion on single icu: pt - {}, visits - {}".format(len(drg_final.SUBJECT_ID.unique()), len(drg_final) ))

    drg_final['stay'] = drg_final.apply(lambda row: "{}_{}".format(row['SUBJECT_ID'], row['HADM_ID']), axis=1)

    drg_final.drop(['DOB', 'ICUSTAY_ID', 'DISCHTIME'], axis=1)

    return drg_final
def filterCohort_mimic4(drg_df, mimic_dir):
    """
        1. filter adult
        2. filter single ICU stay - only one icu stay & no transfer

        return: df of stay information
    """

    icu = pd.read_csv(os.path.join(mimic_dir, 'ICUSTAYS.csv'))
    icu_count = icu.groupby(['hadm_id']).size()
    icu_once = icu.hadm_id.isin(icu_count[icu_count==1].index)
    icu_single = icu[icu_once ][['subject_id', 'hadm_id','intime','los']]
    adm = pd.read_csv(os.path.join(mimic_dir, 'ADMISSIONS.csv'), usecols=['subject_id', 'hadm_id', 'admittime','dischtime'])
    pt = pd.read_csv(os.path.join(mimic_dir, 'PATIENTS.csv'), usecols=['subject_id','anchor_age'])

    drg_adm = pd.merge(drg_df, adm, on=['subject_id', 'hadm_id'])
    drg_dob = pd.merge(drg_adm, pt, on=['subject_id'])

    drg_dob['AGE']  = drg_dob.anchor_age

    drg_adult = drg_dob[drg_dob.AGE >= 18]
    print("age over 18 w/ drg: pt - {}, visits - {}".format(len(drg_adult.subject_id.unique()), len(drg_adult) ))

    drg_final = pd.merge(drg_adult, icu_single, on=['subject_id', 'hadm_id'])
    print("plus criterion on single icu: pt - {}, visits - {}".format(len(drg_final.subject_id.unique()), len(drg_final) ))

    drg_final['stay'] = drg_final.apply(lambda row: "{}_{}".format(row['subject_id'], row['hadm_id']), axis=1)

    drg_final.drop(['dischtime'], axis=1)

    return drg_final
def loadNOTES_mimic3(event_df):
    """
        load noteevents.csv and map to event_df (sub, hadm) and calculate relative note time
    """
    # map notes 
    cols = ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'CATEGORY', 'TEXT']
    print("Loading MIMIC notes...209iter")
    iter_notes = pd.read_csv(os.path.join(args.mimic_dir, 'NOTEEVENTS.csv'), iterator=True, usecols=cols, chunksize=10000)
    dfList = []
    for d in tqdm(iter_notes, total=209):
        dfList.append(event_df.merge(d, on=['SUBJECT_ID', 'HADM_ID']))
    notes_df = pd.concat(dfList)
    print("%d note events in total..." % len(notes_df))
    notes_df = notes_df.dropna(subset=['CHARTTIME'])
    print("%d note events w/ charttime" % len(notes_df))
    intime = notes_df.INTIME.apply(np.datetime64)
    charttime = notes_df.CHARTTIME.apply(np.datetime64)
    diff = (charttime-intime).astype('timedelta64[m]').astype(float)
    notes_df['DIFFTIME'] = diff / 60. 

    return notes_df
def loadNOTES_mimic4(event_df):
    """
        load discharge.csv and map to event_df (sub, hadm) and calculate relative note time
    """
    cols = ['subject_id', 'hadm_id', 'charttime','storetime', 'text']
    print("Loading MIMIC notes...34iter")
    iter_notes = pd.read_csv(os.path.join(args.mimic_dir, 'discharge.csv'), iterator=True, usecols=cols, chunksize=10000)
    dfList = []
    for d in tqdm(iter_notes, total=34):
        dfList.append(event_df.merge(d, on=['subject_id', 'hadm_id']))
    notes_df = pd.concat(dfList)
    print("%d note events in total..." % len(notes_df))
    notes_df = notes_df.dropna(subset=['charttime'])
    notes_df = notes_df.dropna(subset=['storetime'])
    print("%d note events w/ charttime" % len(notes_df))
    intime = notes_df.charttime.apply(np.datetime64)
    charttime = notes_df.storetime.apply(np.datetime64)
    diff = (charttime-intime).astype('timedelta64[m]').astype(float)
    notes_df['DIFFTIME'] = diff / 60. 

    return notes_df
def extract_note_append_adm_diag_mimic3(sub, hadm, adm_diag, notes_df, output_dir):
    """
        extract and save notes for each visit
        get info about note availability
    """
    note_slice = notes_df[(notes_df.SUBJECT_ID == sub) & (notes_df.HADM_ID == hadm)].sort_values('CHARTTIME')
    stay = "{}_{}".format(sub, hadm)
    output_file = stay+'.pk' 

    if len(note_slice) == 0:
        return None

    note_slice = note_slice[['CATEGORY', 'DIFFTIME', 'TEXT']]

    earlies = note_slice.DIFFTIME.iloc[0]
    hour0 = 1 if earlies < 0 else 0
    hour12 = 1 if earlies<12 else 0
    hour24 = 1 if earlies<24 else 0
    hour36 = 1 if earlies<36 else 0
    hours = (hour0, hour12, hour24, hour36)

    valid_mask = note_slice.DIFFTIME <= 48
    valid_df = note_slice[valid_mask]

    if len(valid_df) == 0:
        return None
    else:
        if adm_diag != adm_diag: # check for nan
            adm_diag = ''
        adm_diag_df = pd.DataFrame([{'CATEGORY': 'admission_diag', 'DIFFTIME': -1e+5, 'TEXT': adm_diag}])
        valid_df = pd.concat([adm_diag_df, valid_df], ignore_index=True)
        with open(os.path.join(output_dir, output_file), 'wb') as f:
            pk.dump(valid_df, f, pk.HIGHEST_PROTOCOL)
        return hours
def extract_note_append_adm_diag_mimic4(sub, hadm, notes_df, output_dir):
    """
        extract and save notes for each visit
        get info about note availability
    """
    note_slice = notes_df[(notes_df.subject_id == sub) & (notes_df.hadm_id == hadm)].sort_values('storetime')
    stay = "{}_{}".format(sub, hadm)
    output_file = stay+'.pk' 

    if len(note_slice) == 0:
        return None

    note_slice = note_slice[[ 'DIFFTIME', 'text']]

    earlies = note_slice.DIFFTIME.iloc[0]
    hour0 = 1 if earlies < 0 else 0
    hour12 = 1 if earlies<12 else 0
    hour24 = 1 if earlies<24 else 0
    hour36 = 1 if earlies<36 else 0

    hours = (hour0, hour12, hour24, hour36)

    valid_mask = note_slice.DIFFTIME <= 48
    valid_df = note_slice[valid_mask]

    if len(valid_df) == 0:
        return None
    else:

        with open(os.path.join(output_dir, output_file), 'wb') as f:
            pk.dump(valid_df, f, pk.HIGHEST_PROTOCOL)
        return hours

def split_cohort_mimic3(drg_df, output_dir):
    """
        create train, val, and test split 
        make sure stays of same pt in the single split 
    """

    def split_patients(df, frac=0.1):
        pt_count = df.groupby(['SUBJECT_ID']).size()
        pt_multi = pd.Series(pt_count[pt_count >1].index)
        pt_single= pd.Series(pt_count[pt_count==1].index)

        assert len(pt_single) + len(pt_multi) == len(df.SUBJECT_ID.unique())

        test_multi = pt_multi.sample(frac=frac, random_state=1443)
        test_single = pt_single.sample(frac=frac, random_state=1443)
        test_pt = test_single.append(test_multi)
        test_mask = df.SUBJECT_ID.isin(test_pt)

        test_df = df[test_mask]
        train_df = df[~test_mask]
        

        return train_df, test_df

    test_stays_csv = '%s/test_stays.csv' % output_dir
    if os.path.isfile(test_stays_csv):
        test_stay = pd.read_csv(test_stays_csv)['stay']
        test = drg_df[drg_df['stay'].isin(test_stay)]
        train_val = drg_df[~drg_df['stay'].isin(test_stay)]
    else:
        train_val, test = split_patients(drg_df)

    assert len(train_val)+len(test) == len(drg_df)

    drg_df.to_csv('%s/drg_cohort.csv' % output_dir, index=False)
    train_val.to_csv('%s/train_val.csv' % output_dir, index=False)
    test.to_csv('%s/test.csv' % output_dir, index=False)

    tr_pt, tr_st = len(train_val.SUBJECT_ID.unique()), len(train_val)
    te_pt, te_st = len(test.SUBJECT_ID.unique()), len(test)

    print("..split into train ({} pt, {} st), and test ({} pt, {} st).".format(tr_pt, tr_st, te_pt, te_st))

def split_cohort_mimic4(drg_df, output_dir):
    """
        create train, val, and test split 
        make sure stays of same pt in the single split 
    """

    def split_patients(df, frac=0.1):
        pt_count = df.groupby(['subject_id']).size()
        pt_multi = pd.Series(pt_count[pt_count >1].index)
        pt_single= pd.Series(pt_count[pt_count==1].index)

        assert len(pt_single) + len(pt_multi) == len(df.subject_id.unique())

        test_multi = pt_multi.sample(frac=frac, random_state=1443)
        test_single = pt_single.sample(frac=frac, random_state=1443)
        test_pt = test_single.append(test_multi)
        test_mask = df.subject_id.isin(test_pt)

        test_df = df[test_mask]
        train_df = df[~test_mask]
        

        return train_df, test_df

    test_stays_csv = '%s/test_stays.csv' % output_dir
    if os.path.isfile(test_stays_csv):
        test_stay = pd.read_csv(test_stays_csv)['stay']
        test = drg_df[drg_df['stay'].isin(test_stay)]
        train_val = drg_df[~drg_df['stay'].isin(test_stay)]
    else:
        train_val, test = split_patients(drg_df)

    assert len(train_val)+len(test) == len(drg_df)

    drg_df.to_csv('%s/drg_cohort.csv' % output_dir, index=False)
    train_val.to_csv('%s/train_val.csv' % output_dir, index=False)
    test.to_csv('%s/test.csv' % output_dir, index=False)
    tr_pt, tr_st = len(train_val.subject_id.unique()), len(train_val)
    te_pt, te_st = len(test.subject_id.unique()), len(test)

    print("..split into train ({} pt, {} st), and test ({} pt, {} st).".format(tr_pt, tr_st, te_pt, te_st))


if __name__ == "__main__":
    main()




