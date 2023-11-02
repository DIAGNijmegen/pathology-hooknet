import time
import traceback
from pathlib import Path

import numpy as np
from hooknet.inference.utils import (
    create_lock_file,
    create_output_folders,
    files_exists,
    get_files,
    release_lock_file,
)
from hooknet.inference.writing import create_writers, TILE_SIZE
from tqdm import tqdm
from wholeslidedata.interoperability.asap.masks import MaskType
from wholeslidedata.samplers.utils import crop_data


def execute_inference_single(
    iterator,
    model,
    image_path,
    files,
    output_folder,
    tmp_folder,
):
    print("Init writers...")
    writers = create_writers(
        image_path=image_path,
        files=files,
        output_folder=output_folder,
        tmp_folder=tmp_folder,
    )

    if not writers:
        print(f"Nothing to process for image {image_path}")
        return

    prediction_times = []
    batch_times = []
    print("Applying...")
    index = 0
    batch_time = -1
    for x_batch, y_batch, info in tqdm(iterator):
        if index > 0:
            batch_times.append(time.time() - batch_time)
        x_batch = list(x_batch.transpose(1, 0, 2, 3, 4))
        prediction_time = time.time()
        predictions = model.predict_on_batch(x_batch, argmax=False)
        if index > 0:
            prediction_times.append(time.time() - prediction_time)
            
        for idx, prediction in enumerate(predictions):
            c, r = (
                info["x"] - TILE_SIZE//2,
                info["y"] - TILE_SIZE//2
            )
            mask = crop_data(y_batch[idx][0], model.output_shape[:2])
            for writer in writers:
                writer.write_tile(
                    tile=prediction,
                    coordinates=(int(c), int(r)),
                    mask=mask,
                )
        index += 1
        batch_time = time.time()

    print(f"average batch time: {np.mean(batch_times)}")
    print(f"average prediction time: {np.mean(prediction_times)}")

    # save predictions
    print("Saving...")
    for writer in writers:
        writer.save()


def create_lock_file(lock_file_path):
    print(f"Creating lock file: {lock_file_path}")
    Path(lock_file_path).touch()


def release_lock_file(lock_file_path):
    print(f"Releasing lock file {lock_file_path}")
    Path(lock_file_path).unlink(missing_ok=True)


def create_output_folders(tmp_folder, output_folder):
    output_folder = Path(output_folder)
    tmp_folder = Path(tmp_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    tmp_folder.mkdir(parents=True, exist_ok=True)


def get_files(image_path, model_name, heatmaps):
    files = []
    # prediction
    prediction_file_name = image_path.stem + f"_{model_name}.tif"
    files.append({"name": prediction_file_name, "type": MaskType.PREDICTION})
    # heatmaps
    if heatmaps is not None:
        for value in heatmaps:
            heatmap_file_name = image_path.stem + f"_{model_name}_heat{value}.tif"
            files.append(
                {
                    "name": heatmap_file_name,
                    "type": MaskType.HEATMAP,
                    "heatmap_index": value,
                }
            )

    return files


def files_exists(files, output_folder):
    # check if files alreay exists
    files_exists = []
    for file in files:
        files_exists.append((output_folder / file["name"]).exists())
    return all(files_exists)


def execute_inference(
    image_path,
    model,
    iterator,
    model_name,
    output_folder,
    tmp_folder,
    heatmaps,
):
    image_path = Path(image_path)
    output_folder = Path(output_folder)
    tmp_folder = Path(tmp_folder)

    print("Create output folder")
    create_output_folders(tmp_folder=tmp_folder, output_folder=output_folder)
    lock_file_path = output_folder / (image_path.stem + f"{model_name}.lock")
    if lock_file_path.exists():
        print("Lock file exists, skipping inference.")
        return

    files = get_files(image_path=image_path, model_name=model_name, heatmaps=heatmaps)

    if files_exists(files=files, output_folder=output_folder):
        print(f"All output files already exist, skipping inference.")
        return

    try:
        create_lock_file(lock_file_path=lock_file_path)
        print("Run inference")
        execute_inference_single(
            iterator=iterator,
            model=model,
            image_path=image_path,
            files=files,
            output_folder=output_folder,
            tmp_folder=tmp_folder,
        )
        print("Stopping iterator")
        iterator.stop()

    except Exception as e:
        print("Exception")
        print(e)
        print(traceback.format_exc())
    finally:
        release_lock_file(lock_file_path=lock_file_path)
    return files
