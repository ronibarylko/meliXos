#!/bin/bash

python3 reader.py eci2019nlp/snli_1.0_train_filtered.jsonl eci2019nlp/snli_1.0_train_gold_labels.csv
echo "Train data parsed correctly"
mv data.txt train.txt
python3 reader.py eci2019nlp/snli_1.0_test_filtered.jsonl
echo "Predict data parsed correctly"
mv data.txt predict.txt
python3 main.py train.txt predict.txt 'predict'
