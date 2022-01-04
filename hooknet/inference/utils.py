from pathlib import Path
from hooknet.inference.writing import MaskType

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
            files.append({"name": heatmap_file_name, "type": MaskType.HEATMAP, "heatmap_index": value})

    return files


def files_exists(files, output_folder):
    # check if files alreay exists
    files_exists = []
    for file in files:
        files_exists.append((output_folder / file["name"]).exists())
    return all(files_exists)
