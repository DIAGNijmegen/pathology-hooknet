from pathlib import Path

from tqdm.notebook import tqdm
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.image.wholeslideimagewriter import (
    HeatmapTileCallback,
    PredictionTileCallback,
    WholeSlideMaskWriter,
)
from wholeslidedata.iterators import create_batch_iterator
from wholeslidedata.source.configuration.config import (
    get_paths,
    insert_paths_into_config,
)

from hooknet.configuration.config import create_hooknet
import argparse
from pathlib import Path
from shutil import copy

SPACING = 0.5
TILE_SIZE = 1024
OUTPUT_SIZE = 1030


def _create_lock_file(lock_path):
    Path(lock_path).touch()


def _release_lock_file(lock_path):
    Path(lock_path).unlink(missing_ok=True)


def _copy_temp_path_to_output_path(output_paths_tmp, output_paths):
    for output_path_tmp, output_path in zip(output_paths_tmp, output_paths):
       copy(output_path_tmp, output_path)

def _init_writers(image_path, output_folder, tmp_folder, model_name, heatmaps):
    output_paths_tmp = []
    output_paths = []
    writers = {}

    # get info
    with WholeSlideImage(image_path) as wsi:
        shape = wsi.shapes[wsi.get_level_from_spacing(SPACING)]
        real_spacing = wsi.get_real_spacing(SPACING)

    # prediction
    prediction_file_name = image_path.stem + f"_{model_name}.tif"
    if not (output_folder / prediction_file_name).exists():
        output_paths.append(output_folder / prediction_file_name)
        output_paths_tmp.append(tmp_folder / prediction_file_name)
        writers["prediction"] = WholeSlideMaskWriter(
            callbacks=(PredictionTileCallback(),)
        )
        writers["prediction"].write(
            path=tmp_folder / prediction_file_name,
            spacing=real_spacing,
            dimensions=shape,
            tile_shape=(TILE_SIZE, TILE_SIZE),
        )

    # heatmaps
    if heatmaps is not None:
        for value in heatmaps:
            heatmap_file_name = image_path.stem + f"_{model_name}_heat{value}.tif"
            if not (output_folder / heatmap_file_name).exists():
                output_paths.append(output_folder / heatmap_file_name)
                output_paths_tmp.append(tmp_folder / heatmap_file_name)

                writers[value] = WholeSlideMaskWriter(
                    callbacks=(HeatmapTileCallback(heatmap_index=value),)
                )
                writers[value].write(
                    path=tmp_folder / heatmap_file_name,
                    spacing=real_spacing,
                    dimensions=shape,
                    tile_shape=(TILE_SIZE, TILE_SIZE),
                )
    return writers, output_paths_tmp, output_paths


def _apply_single_inference(
    iterator,
    model,
    image_path,
    output_folder,
    tmp_folder,
    model_name,
    heatmaps,
):

    writers, output_paths_tmp, output_paths = _init_writers(
        image_path=image_path,
        output_folder=output_folder,
        tmp_folder=tmp_folder,
        model_name=model_name,
        heatmaps=heatmaps,
    )

    for x_batch, y_batch, info in tqdm(iterator):
        x_batch = list(x_batch.transpose(1, 0, 2, 3, 4))
        predictions = model.predict_on_batch(x_batch, argmax=False)

        for idx, prediction in enumerate(predictions):
            point = info["sample_references"][idx]["point"]
            c, r = point.x - OUTPUT_SIZE // 4, point.y - OUTPUT_SIZE // 4

            for writer in writers.values():
                writer.write_tile(
                    tile=prediction,
                    coordinates=(int(c), int(r)),
                    mask=y_batch[idx][0],
                )

    # save predictions
    for writer in writers.values():
        writer.save()

    return output_paths_tmp, output_paths


def apply(
    user_config,
    mode,
    model_name,
    output_folder,
    tmp_folder,
    cpus,
    source_preset,
    heatmaps,
):
    output_paths_tmp = []
    output_paths = []

    # create folders
    output_folder = Path(output_folder)
    tmp_folder = Path(tmp_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    tmp_folder.mkdir(parents=True, exist_ok=True)

    model = create_hooknet(user_config=user_config)

    for image_path, annotation_path in get_paths(user_config, preset=source_preset):
        # do something with lockfile image_path
        # create lock file path
        lock_file_path = output_folder / image_path.stem + ".lock"

        if lock_file_path.exists():
            continue

        try:
            _create_lock_file

            # Create iterator
            user_config = insert_paths_into_config(
                user_config, image_path, annotation_path
            )
            iterator = create_batch_iterator(
                mode=mode,
                user_config=user_config,
                presets=(
                    "files",
                    "slidingwindow",
                ),
                cpus=cpus,
                number_of_batches=-1,
                search_paths=(str(Path(user_config).parent),),
            )

            output_paths_tmp, output_paths = _apply_single_inference(
                iterator=iterator,
                model=model,
                user_config=user_config,
                image_path=image_path,
                output_folder=output_folder,
                tmp_folder=tmp_folder,
                model_name=model_name,
                heatmaps=heatmaps,
            )

        except Exception as exception:
            print(exception)
        else:
            _copy_temp_path_to_output_path(output_paths_tmp, output_paths)
        finally:
            iterator.stop()
            _release_lock_file(lock_file_path)


def _parse_args():
    # create argument parser
    argument_parser = argparse.ArgumentParser(description="Experiment")
    argument_parser.add_argument("-u", "--user_config", required=True)
    argument_parser.add_argument("-n", "--model_name", required=True)
    argument_parser.add_argument("-o", "--output_folder", required=True)
    argument_parser.add_argument("-d", "--tmp_folder", required=True)
    argument_parser.add_argument("-m", "--mode", required=False)
    argument_parser.add_argument("-s", "--source_preset", required=False)
    argument_parser.add_argument("-c", "--cpus", required=False)
    argument_parser.add_argument("-t", "--heatmaps", nargs="+", required=False)
    args = vars(argument_parser.parse_args())

    if "mode" not in args or not args['mode']:
        args["mode"] = "default"

    if "source_preset" not in args or not args['source_preset']:
        args["source_preset"] = "folders"
    else:
        args["source_preset"] = args["source_preset"]

    if "cpus" not in args or not args["cpus"]:
        args["cpus"] = 1

    if "heatmaps" not in args or not args['heatmaps']:
        args["heatmaps"] = None

    return args


def main():
    args = _parse_args()
    apply(
        user_config=args["user_config"],
        mode=args["user_config"],
        model_name=args["model_name"],
        output_folder=args["output_folder"],
        tmp_folder=args["tmp_folder"],
        cpus=args["cpus"],
        source_preset=args["source_preset"],
        heatmaps=args["heatmaps"],
    )


if __name__ == "__main__":
    main()