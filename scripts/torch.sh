 #! /bin/bash
 
 python3 -m experimart.training.train \
    experimart.default.log_path="'./'" \
    experimart.default.project="'hooknet-torch-test'" \
    experimart.default.data_settings.cpus=12 \
    -c /home/user/pathology-hooknet/hooknet/training/configuration/torchtraining.yml \
    -p wandb/tracker.yml \
    -p torch/training.yml \
    -p wholeslidedata/dataiterator.yml \ 
    