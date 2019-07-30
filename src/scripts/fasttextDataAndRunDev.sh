#!/bin/bash
RED="\033[1;31m"
NOCOLOR="\033[0m"

python3 reader.py eci2019nlp/snli_1.0_train_filtered.jsonl eci2019nlp/snli_1.0_train_gold_labels.csv
echo -e "${RED}Train data parsed correctly${NOCOLOR}"
mv data.txt train.txt
python3 reader.py eci2019nlp/snli_1.0_dev_filtered.jsonl eci2019nlp/snli_1.0_dev_gold_labels.csv
echo -e "${RED}Dev data parsed correctly${NOCOLOR}"
mv data.txt dev.txt
python3 fasttext_model.py train.txt dev.txt 'test'
echo -e "${RED}Result was produced correctly${NOCOLOR}"
rm *.txt
echo -e "${RED}Temporary files removed${NOCOLOR}"
