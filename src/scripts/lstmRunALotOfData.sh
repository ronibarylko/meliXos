#!/bin/bash
rm lstm.csv
printf "lr,embedd,hidden,batch,epoch,firstRun,secondRun,thirdRun\n" >> lstm.csv
while IFS=, read -r lr	embedd	hidden batch	epoch data
do
  declare -a corridas
  for i in 0 1 2
  do
    corridas[i]=$(python3 lstm_model.py --function 'test' --learning $lr --embedding $embedd --hidden $hidden --batch $batch --epoch $epoch --data $data)
    echo "Resultado ${corridas[i]}"
  done
  printf "$lr,$embedd,$hidden,$batch,$epoch,$data,${corridas[0]},${corridas[1]},${corridas[2]}\n" >> lstm.csv
done < runData.csv
