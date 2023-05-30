 #! /bin/bash
 
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

 python3 -m experimart.training.train \
    experimart.default.log_path="'./'" \
    experimart.default.project="'hooknet-torch-test'" \
    experimart.default.data_settings.cpus=16 \
    -c $SCRIPT_DIR/configs/torchtraining.yml \
    -p wandb/tracker.yml \
    -p torch/training.yml \
    -p wholeslidedata/dataiterator.yml \ 
    