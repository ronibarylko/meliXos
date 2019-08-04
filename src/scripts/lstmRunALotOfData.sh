#!/bin/bash
RED="\033[1;31m"
NOCOLOR="\033[0m"

while IFS=, read -r lr	embedd	hidden batch	epoch
do
  echo -e "${RED}LR: $lr, EMBEDD: $embedd, HIDDEN: $hidden, BATCH: $batch, EPOCH: $epoch ${NOCOLOR}"
  python3 lstm_model.py --function 'test' --learning $lr --embedding $embedd --hidden $hidden --batch $batch --epoch $epoch
done < runData.csv
