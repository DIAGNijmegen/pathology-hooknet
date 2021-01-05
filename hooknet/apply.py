import math
import os
import os.path
import pathlib
import shutil

import cv2
import numpy as np
import yaml
import json

from argconfigparser.argconfigparser import ArgumentConfigParser
from source.image.imagereader import ImageReader
from source.inference import Inference
from source.model import HookNet


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


config = ArgumentConfigParser(
    os.path.join(pathlib.Path(__file__).parent, "apply_parameters.yml")
).parse_args()

print(config)


# initialize model
hooknet = HookNet(
    input_shape=config["input_shape"],
    n_classes=config["n_classes"],
    hook_indexes=config["hook_indexes"],
    depth=config["depth"],
    n_convs=config["n_convs"],
    filter_size=config["filter_size"],
    n_filters=config["n_filters"],
    padding=config["padding"],
    batch_norm=config["batch_norm"],
    activation=config["activation"],
    learning_rate=config["learning_rate"],
    opt_name=config["opt_name"],
    l2_lambda=config["l2_lambda"],
    loss_weights=config["loss_weights"],
    merge_type=config["merge_type"],
)


# load weights
hooknet.load_weights(config["weights_path"])

image_path = config["image_path"]
mask_path = config["mask_path"]
output_path = config["output_path"]

output_file = (
    os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0])
    + "_hooknet.tif"
)


prediction_path = output_file

print("image_path:", image_path)
print("mask_path:", mask_path)
print("output_path:", output_path)


if config["copy"] and "work_dir" in config:
    print("copy to local folder...")
    shutil.copy2(image_path, config["work_dir"])

    if mask_path:
        shutil.copy2(mask_path, config["work_dir"])

    image_path = os.path.join(config["work_dir"], os.path.basename(image_path))

    mask_path = (
        os.path.join(config["work_dir"], os.path.basename(mask_path))
        if mask_path
        else mask_path
    )

    output_file = (
        os.path.join(
            config["work_dir"], os.path.splitext(os.path.basename(image_path))[0]
        )
        + "_hooknet.tif"
    )

print("apply hooknet...")
apply = Inference(
    wsi_path=config["image_path"],
    mask_path=config["mask_path"],
    output_file=output_file,
    input_shape=config["input_shape"],
    output_shape=hooknet.output_shape,
    resolutions=config["resolutions"],
    batch_size=config["batch_size"],
    cpus=config["cpus"],
    queue_size=config["queue_size"],
    model_instance=hooknet,
    multi_loss=config["multi_loss"],
    mask_ratio=config["mask_ratio"],
)
apply.start()

if config["copy"] and "work_dir" in config:
    print("copy result...")
    shutil.copy2(output_file, config["output_path"])
    print("done.")

if config["calc_score"]:
    print("Calculating score..")
    img_reader = ImageReader(output_file, 0.2)
    ratio = 32
    spacing = img_reader.spacings[0] * 2 ** math.log(ratio, 2)
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) / 255
    patch = img_reader.content(spacing)
    unique, counts = np.unique(patch, return_counts=True)
    score = dict(list(zip(map(int, unique), map(int, counts))))

    output = {
        "image_path": config["image_path"],
        "mask_path": config["mask_path"],
        "prediction_path": prediction_path,
        "score": score,
    }

    yaml_output_file = (
        os.path.join(
            config["output_path"], os.path.splitext(os.path.basename(image_path))[0]
        )
        + "_output.yml"
    )

    with open(yaml_output_file, "w") as outfile:
        yaml.dump(output, outfile, default_flow_style=False)

else:
    output = [
        (
            {
                "entity": config["image_path"],
                "output": f"filepath:images/{os.path.basename(prediction_path)}",
                "error_messages": [],
                "metrics": {"f1": "N/A"},
            }
        )
    ]

    with open("/home/user/results.json", "w") as file:
        json.dump(output, file)
