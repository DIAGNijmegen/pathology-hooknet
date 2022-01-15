import time
import traceback
from pathlib import Path
import numpy as np
from hooknet.configuration.config import create_hooknet
from hooknet.inference.utils import (
    create_lock_file,
    create_output_folders,
    files_exists,
    get_files,
    release_lock_file,
)
from hooknet.inference.writing import create_writers
from tqdm import tqdm
from wholeslidedata.iterators import create_batch_iterator
from wholeslidedata.source.configuration.config import (
    get_paths,
    insert_paths_into_config,
)


def _execute_inference_single(
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
            batch_times.append(time.time()-batch_time)
        x_batch = list(x_batch.transpose(1, 0, 2, 3, 4))
        prediction_time = time.time()
        predictions = model.predict_on_batch(x_batch, argmax=False)
        if index > 0:
            prediction_times.append(time.time()-prediction_time)

        for idx, prediction in enumerate(predictions):
            point = info["sample_references"][idx]["point"]
            c, r = (
                point.x - model.output_shape[1] // 4,
                point.y - model.output_shape[0] // 4,
            )

            for writer in writers:
                writer.write_tile(
                    tile=prediction,
                    coordinates=(int(c), int(r)),
                    mask=y_batch[idx][0],
                )
        index += 1
        batch_time = time.time()

    print(f"average batch time: {np.mean(batch_times)}")
    print(f"average prediction time: {np.mean(prediction_times)}")
    
    # save predictions
    print("Saving...")
    for writer in writers:
        writer.save()


def execute_inference(
    user_config,
    mode,
    model_name,
    output_folder,
    tmp_folder,
    cpus,
    source_preset,
    heatmaps,
):
    print("Create model")
    model = create_hooknet(user_config=user_config)

    print("Create output folder")
    create_output_folders(tmp_folder=tmp_folder, output_folder=output_folder)

    for image_path, annotation_path in get_paths(user_config, preset=source_preset):
        print(f"PROCESSING: {image_path}, with {annotation_path}....")

        lock_file_path = output_folder / (image_path.stem + ".lock")
        if lock_file_path.exists():
            print("Lock file exists, skipping inference.")
            continue

        files = get_files(
            image_path=image_path, model_name=model_name, heatmaps=heatmaps
        )

        if files_exists(files=files, output_folder=output_folder):
            print(f"All output files already exist, skipping inference.")
            continue

        try:
            create_lock_file(lock_file_path=lock_file_path)

            # Create iterator
            print("Creating iterator...")
            user_config_dict = insert_paths_into_config(
                user_config, image_path, annotation_path
            )
            iterator = create_batch_iterator(
                mode=mode,
                user_config=user_config_dict["wholeslidedata"],
                presets=(
                    "files",
                    "slidingwindow",
                ),
                cpus=cpus,
                number_of_batches=-1,
                search_paths=(str(Path(user_config).parent),),
            )
            
            print("Run inference")
            _execute_inference_single(
                iterator=iterator,
                model=model,
                image_path=image_path,
                files=files,
                output_folder=output_folder,
                tmp_folder=tmp_folder,
            )
            print('Stopping iterator')
            iterator.stop()

        except Exception as e:
            print("Exception")
            print(e)
            print(traceback.format_exc())
        finally:
            release_lock_file(lock_file_path=lock_file_path)
        print("--------------")
