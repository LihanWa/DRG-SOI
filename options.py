import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--collectText", default='No', type=str, choices=['Yes','No'], help="If text_raw is already created, it can be set as 'No', then only text_tokens or text_LongFormer will be created.")
parser.add_argument("--LongFormer", default='No', type=str, choices=['Yes','No'], help="If you want to create data for LongFormer, it should be set as Ture")
parser.add_argument("--use_AbbFull", default='No', type=str, choices=['Yes','No'], help="If you want to use my abbreviation-fullspelling.txt when creating data for training LongFormer, it should be set as Ture")
parser.add_argument("--data_dir", default='data', type=str, help="Directory for data folder which contains root data, as well as subfolders for text files")
parser.add_argument("--mimic_dir", default='~/mimic/', type=str, help="Dir for MIMIC-III or MIMIC-IV")
parser.add_argument("--wd", default=1e-4, type=float)
parser.add_argument("--data_source", default='ms', type=str, choices=['ms', 'apr_mimic3','apr_mimic4'])
parser.add_argument("--rule_dir", default='rules', type=str, help="path for officials `rules' for DRG grouping and weights")
parser.add_argument("--dropout_rate", default=0.3, type=float)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--epochs", default=36, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--device", default='0', type=str)
parser.add_argument("--eval_model", type=str, choices=['train','eval'])
args, unknown = parser.parse_known_args()


