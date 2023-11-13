# DRG-SOI
BioWordVec_PubMed_MIMICIII_d200.vec.bin
The [BioWordVec](https://github.com/ncbi-nlp/BioSentVec) word embedding is used in the experiments. 
```shell
# create cohort -- will print the cohort statistics during processing
python create_cohort.py --data_dir $DATA_PATH --mimic_dir $MIMIC_PATH --threshold 48
```

<p align="center" width="100%">
<img src="flow.jpg" width=600 >
</p>
