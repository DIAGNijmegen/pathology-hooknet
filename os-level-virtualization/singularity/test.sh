#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 12:00:00

set -Eeuo pipefail

DOCKER_USERNAME=...
DOCKER_PASSWORD=...
DOCKER_IMAGE="martvanrijthoven/private:hooknet"
COMMAND="python -u /usr/local/lib/python3.6/dist-packages/hooknet/apply.py --weights_path /host_data/colon_weights.h5 --image_path /host_data/19-6440.tif --mask_path /host_data/19-6440_mask.png --mask_ratio 32 --output_path /host_data/"

export SINGULARITY_DOCKER_USERNAME=${DOCKER_USERNAME}
export SINGULARITY_DOCKER_PASSWORD=${DOCKER_PASSWORD}
export SINGULARITY_BINDPATH="/home/mart/MIT-docker-data:/host_data"


singularity exec --nv --cleanenv --no-home  --containall docker://${DOCKER_IMAGE} ${COMMAND}
