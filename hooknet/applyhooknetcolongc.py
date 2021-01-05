from hooknet.libs.experiment.experiment.apply import Apply
from hooknet.libs.experiment.models.hooknet import Hooknet
import glob
import os
import sys
import json


# settings
model_name = "hooknet"
patch_shape = [1244, 1244, 3]
padding = "valid"
depth = 4
n_filters = 64
learning_rate = 0.05
batch_norm = True
l2_lambda = 0.001
opt_name = "adam"
labels = ["adenomatous-polyp-lgd", "adenomatous-polyp-hgd", "non-informative"]
hook_indexes = [3, 3]
multi_loss = False

# settings
batch_size = 1
cpus = 6
queue_size = 10
input_shape = [1244, 1244, 3]
output_shape = [1030, 1030, 4]
resolutions = [0.25, 4.0]
mask_ratio = 32

hooknet = Hooknet(
    model_name,
    patch_shape,
    padding,
    depth,
    n_filters,
    learning_rate,
    batch_norm,
    l2_lambda,
    opt_name,
    labels,
    hook_indexes,
    multi_loss,
)

weights = "/home/user/model.h5"
hooknet._model_instance.load_weights(weights)

images = []
for ext in ("*.tif", "*.tiff", "*.mrxs", "*.ndpi"):
    images.extend(glob.glob(os.path.join("/home/user/data/input", ext)))

masks = []
for ext in ("*.tif", "*.tiff", "*.mrxs", "*.ndpi"):
    masks.extend(glob.glob(os.path.join("/home/user/data/bgmask/", ext)))

output = []
for wsi_path in images:
    image_name = os.path.splitext(os.path.basename(wsi_path))[0]
    mask_path = None

    paired_masks = [mask for mask in masks if image_name in mask]
    if len(paired_masks) > 0:
        mask_path = paired_masks[0]
    else:
        print("no mask for: ", image_name)
        print("skipping...", image_name)
        continue

    output_path = os.path.join(
        "/home/user/data/result", image_name + "_hooknet_prediction.tif"
    )

    print("wsi path:", wsi_path)
    print("mask path:", mask_path)
    print("output_path:", output_path)
    error = None
    try:
        apply = Apply(
            wsi_path,
            mask_path,
            output_path,
            input_shape,
            output_shape,
            resolutions,
            batch_size,
            cpus,
            queue_size,
            hooknet._model_instance,
            multi_loss,
            mask_ratio,
        )
        apply.start()
    except Exception as e:
        # Output error message, and store for returning in the output file
        print(str(e))
        error = str(e)

    output.append(
        {
            "entity": wsi_path,
            "output": f"filepath:images/{os.path.basename(output_path)}"
            if not error
            else None,
            "error_messages": [error] if error else [],
            "metrics": {"f1": "N/A"},
        }
    )

    with open("/home/user/results.json", "w") as file:
        json.dump(output, file)
