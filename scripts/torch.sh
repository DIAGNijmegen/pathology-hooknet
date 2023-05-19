 #! /bin/bash
 
 python3 -m experimart.training.train \
    experimart.default.log_path="'./'" \
    experimart.default.project="'hooknet-torch-test'" \
    -c /home/user/pathology-hooknet/hooknet/configuration/training.yml \
    -p wandb/tracker.yml \
    -p torch/training.yml \
    -p wholeslidedata/dataiterator.yml \ 
    