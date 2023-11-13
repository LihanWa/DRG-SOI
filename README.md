# DRG-SOI
## Checkpoints:
If you want to run our result directly,firstly, you should finish running step 1-3. Then, you can download the checkpoints from the links and move it to the root directory. Then, the "--eval_model" should be set as eval.

checkpoint for MS-DRG of MIMIC-III of LGLC: [bestMS](https://drive.google.com/file/d/1I-XlJP0Gj3GK6U4ebRpxn0PwtHdZIoK4/view?usp=sharing)

checkpoint for APR-DRG of MIMIC-III of LGLC: [bestAPR_MIMIC3](https://drive.google.com/file/d/1-QrKJ2wR5fHsMxMZVhoE_AZD46kRR8zx/view?usp=sharing)

checkpoint for APR-DRG of MIMIC-IV of LGLC: [bestAPR_MIMIC4](https://drive.google.com/file/d/1-RQsxiwJGNa2sPZmJASfs1trHsG6Tw3E/view?usp=sharing)

checkpoint for MS-DRG of MIMIC-III of LongFormer: [bestMS_LongFormer](https://drive.google.com/file/d/1-If6pLWlqAPEEkc0lE_eC07dCDaRkjc1/view?usp=sharing)

checkpoint for APR-DRG of MIMIC-III of LongFormer: [bestAPR_MIMIC3_LongFormer](https://drive.google.com/file/d/1--IU-v4MxLvD7aaRw2oHjRFKTolInfK_/view?usp=sharing)

checkpoint for APR-DRG of MIMIC-IV of LongFormer: [bestAPR_MIMIC4_LongFormer](https://drive.google.com/file/d/1Jc1auu8L9nGKyAlP3p7YoGs9HtYJDVvB/view?usp=sharing)


BioWordVec_PubMed_MIMICIII_d200.vec.bin
The [BioWordVec](https://github.com/ncbi-nlp/BioSentVec) word embedding is used in the experiments. 
```shell
# create cohort -- will print the cohort statistics during processing
python create_cohort.py --data_dir $DATA_PATH --mimic_dir $MIMIC_PATH --threshold 48
```

<p align="center" width="100%">
<img src="flow.jpg" width=600 >
</p>
