#!/bin/bash

SCRIPT=$1
COMPUTER=$(hostname)
let NPROC=$(echo $$)+3 # dovrebbe essere esattamente il PID del nohup
LOG=${COMPUTER}_${NPROC}_${SCRIPT::-3}.log

echo $LOG
nohup python3 -u ./$SCRIPT &> $LOG &

# Es. ~$: bash run:backgroun.sh nome-script.py
# computer_nproc_nome-script.log

