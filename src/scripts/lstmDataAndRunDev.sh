#!/bin/bash
RED="\033[1;31m"
NOCOLOR="\033[0m"

python3 reader.py eci2019nlp/snli_1.0_train_filtered.jsonl eci2019nlp/snli_1.0_train_gold_labels.csv
echo -e "${RED}Train data parsed correctly${NOCOLOR}"
mv data.txt train_data.txt
python3 reader.py eci2019nlp/snli_1.0_dev_filtered.jsonl eci2019nlp/snli_1.0_dev_gold_labels.csv
echo -e "${RED}Dev data parsed correctly${NOCOLOR}"
mv data.txt dev_data.txt
python3 reader.py eci2019nlp/snli_1.0_test_filtered.jsonl
echo -e "${RED}Test sentences parsed correctly${NOCOLOR}"
mv data.txt test_sentences.txt
python3 reader.py eci2019nlp/snli_1.0_train_filtered.jsonl
echo -e "${RED}Train sentences parsed correctly${NOCOLOR}"
mv data.txt train_sentences.txt
python3 reader.py eci2019nlp/snli_1.0_dev_filtered.jsonl
echo -e "${RED}Dev sentences parsed correctly${NOCOLOR}"
mv data.txt dev_sentences.txt
python3 lstm_model.py --function 'test' --logging True
echo -e "${RED}Result was produced correctly${NOCOLOR}"
rm *.txt
echo -e "${RED}Temporary files removed${NOCOLOR}"
