# DRG-SOI
## Checkpoints:
To run our results directly: Firstly, you should finish running step 1-3. Then, you can download the checkpoints from the links and move it to the root directory. 

The "--eval_model" should be set as eval.

If you want to run LongFormer, the "--LongFormer" should be set as "Yes"; otherwise, it should be set as "No". If running LongFormer, the "--batch_size" depends on your GPU and RAM. I set it as 3.

The "--data_source" can be set as "ms", or "apr_mimic3", or "apr_mimic4", which represents MS-DRG and APR-DRG of MIMIC-III, and APR-DRG of MIMIC-IV respectively.

```shell
!python main.py --epochs 1 --patience 10 --lr 1.2e-4 --wd 0 --data_source apr_mimic3 --eval_model eval --LongFormer Yes --batch_size 3
```

checkpoint for MS-DRG of MIMIC-III of LGLC: [bestMS](https://drive.google.com/file/d/1I-XlJP0Gj3GK6U4ebRpxn0PwtHdZIoK4/view?usp=sharing)

checkpoint for APR-DRG of MIMIC-III of LGLC: [bestAPR_MIMIC3](https://drive.google.com/file/d/1-QrKJ2wR5fHsMxMZVhoE_AZD46kRR8zx/view?usp=sharing)

checkpoint for APR-DRG of MIMIC-IV of LGLC: [bestAPR_MIMIC4](https://drive.google.com/file/d/1-RQsxiwJGNa2sPZmJASfs1trHsG6Tw3E/view?usp=sharing)

checkpoint for MS-DRG of MIMIC-III of LongFormer: [bestMS_LongFormer](https://drive.google.com/file/d/1-If6pLWlqAPEEkc0lE_eC07dCDaRkjc1/view?usp=sharing)

checkpoint for APR-DRG of MIMIC-III of LongFormer: [bestAPR_MIMIC3_LongFormer](https://drive.google.com/file/d/1--IU-v4MxLvD7aaRw2oHjRFKTolInfK_/view?usp=sharing)

checkpoint for APR-DRG of MIMIC-IV of LongFormer: [bestAPR_MIMIC4_LongFormer](https://drive.google.com/file/d/1Jc1auu8L9nGKyAlP3p7YoGs9HtYJDVvB/view?usp=sharing)

###1.Create dataset and preprocess:
To get files of  
```shell
# 
!python data_preprocess.py --data_dir data --mimic_dir /Users/lihan/Downloads --threshold 48 --data_source ms --LongFormer Yes --collectText No --use_AbbFull No
```
BioWordVec_PubMed_MIMICIII_d200.vec.bin
The [BioWordVec](https://github.com/ncbi-nlp/BioSentVec) word embedding is used in the experiments. 
```shell
# create cohort -- will print the cohort statistics during processing
python create_cohort.py --data_dir $DATA_PATH --mimic_dir $MIMIC_PATH --threshold 48
```

<p align="center" width="100%">
<img src="flow.jpg" width=600 >
</p>
