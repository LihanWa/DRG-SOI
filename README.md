# DRG-SOI
#### 1.Our code is for MS-DRG and APR-DRG of MIMIC-III, and APR-DRG of MIMIC-IV, by setting "--data_source". 

#### 2.Our code is for our model (LGLC) and Clinical LongFormer by setting "--LongFormer". 

#### 3. For Clinical LongFormer, you can choose to use our abbreviation_fullspelling.txt file to replace abbreviation in the clinical texts with fullspelling, by setting "--use_AbbFull".

#### 4. Our code is for both training and testing by setting "--eval_model".
## 1.combine information from files and preprocess:
Get files for MS-DRG and APR-DRG of MIMIC-III: 

DRGCODES.csv, ICUSTAYS.csv, ADMISSIONS.csv, PATIENTS.csv, and NOTEEVENTS.csv are available in [MIMIC_III](https://physionet.org/content/mimiciii/1.4/).

Get files for APR-DRG of MIMIC-IV: 

DRGCODES.csv, ICUSTAYS.csv, ADMISSIONS.csv, and PATIENTS.csv are available in [MIMIC_IV](https://physionet.org/content/mimiciv/2.2/). discharge.csv is available in [discharge](https://physionet.org/content/mimic-iv-note/2.2/note/#files-panel).

The method of combining information from files is the same whether it is for LongFormer or not. "--collectText" can be set as "No", if "text_raw" file is already created. For data preprocess, if you want to preprocess LongFormer, the "--LongFormer" should be set as "Yes"; otherwise, it should be set as "No". 
```shell

!python data_preprocess.py --data_dir data --mimic_dir  --threshold 48 --data_source ms --LongFormer Yes --collectText No --use_AbbFull No
```
BioWordVec_PubMed_MIMICIII_d200.vec.bin
The [BioWordVec](https://github.com/ncbi-nlp/BioSentVec) word embedding is used in the experiments. 
## 2.create dataset
```shell
!python create_dataset.py  --data_source ms --LongFormer Yes --use_AbbFull No
```

## Eval:

To run our results directly: Firstly, you should finish running step 1-2. Then, you can download the checkpoints from the links and move it to the root directory. 

The "--eval_model" should be set as eval.

If you want to run LongFormer, the "--LongFormer" should be set as "Yes"; otherwise, it should be set as "No". If running LongFormer, the "--batch_size" depends on your GPU RAM and System RAM. I set it as 3.

The "--data_source" can be set as "ms", or "apr_mimic3", or "apr_mimic4", which represents MS-DRG and APR-DRG of MIMIC-III, and APR-DRG of MIMIC-IV respectively.

```shell
#If you want to train by yourself, "--eval_model" should be set as "train".
!python main.py --epochs 1 --patience 10 --lr 1.2e-4 --wd 0 --data_source apr_mimic3 --eval_model eval --LongFormer Yes --batch_size 3
```
### checkpoints:
checkpoint for MS-DRG of MIMIC-III of LGLC: [bestMS](https://drive.google.com/file/d/1I-XlJP0Gj3GK6U4ebRpxn0PwtHdZIoK4/view?usp=sharing)

checkpoint for APR-DRG of MIMIC-III of LGLC: [bestAPR_MIMIC3](https://drive.google.com/file/d/1-QrKJ2wR5fHsMxMZVhoE_AZD46kRR8zx/view?usp=sharing)

checkpoint for APR-DRG of MIMIC-IV of LGLC: [bestAPR_MIMIC4](https://drive.google.com/file/d/1-RQsxiwJGNa2sPZmJASfs1trHsG6Tw3E/view?usp=sharing)

checkpoint for MS-DRG of MIMIC-III of LongFormer: [bestMS_LongFormer](https://drive.google.com/file/d/1-If6pLWlqAPEEkc0lE_eC07dCDaRkjc1/view?usp=sharing)

checkpoint for APR-DRG of MIMIC-III of LongFormer: [bestAPR_MIMIC3_LongFormer](https://drive.google.com/file/d/1--IU-v4MxLvD7aaRw2oHjRFKTolInfK_/view?usp=sharing)

checkpoint for APR-DRG of MIMIC-IV of LongFormer: [bestAPR_MIMIC4_LongFormer](https://drive.google.com/file/d/1Jc1auu8L9nGKyAlP3p7YoGs9HtYJDVvB/view?usp=sharing)

