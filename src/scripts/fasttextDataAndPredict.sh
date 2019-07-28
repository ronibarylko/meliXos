#!/bin/bash
RED="\033[1;31m"
NOCOLOR="\033[0m"

python3 reader.py eci2019nlp/snli_1.0_train_filtered.jsonl eci2019nlp/snli_1.0_train_gold_labels.csv
echo -e "${RED}Train data parsed correctly${NOCOLOR}"
mv data.txt train.txt
python3 reader.py eci2019nlp/snli_1.0_test_filtered.jsonl
echo -e "${RED}Predict data parsed correctly${NOCOLOR}"
mv data.txt predict.txt
python3 fasttext.py train.txt predict.txt 'predict'
python3 generate_answer.py
echo -e "${RED}Result was produced correctly${NOCOLOR}"
rm *.txt
echo -e "${RED}Temporary files removed${NOCOLOR}"
