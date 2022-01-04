from enum import Enum, auto
from shutil import copyfile
from pathlib import Path

from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.image.wholeslideimagewriter import (HeatmapTileCallback,
                                                        PredictionTileCallback,
                                                        WholeSlideMaskWriter)

SPACING = 0.5
TILE_SIZE = 1024


class MaskType(Enum):
    """Different mask types
    
    The PREDICTION type is for writing masks with prediction values, range=(0, num_classes)
    The HEATMAP type is for writing masks with heatmap values, range=(0, 255)
    """

    
    PREDICTION = auto()    
    HEATMAP = auto()


class TmpWholeSlideMaskWriter(WholeSlideMaskWriter):
    

    def __init__(self, output_path: Path, callbacks=(), suffix='.tif'):
        """Writes temp file and copies the tmp file to an output folder in the save method.

        Args:
            output_path (Path): path to copy the writed file when saving.
        """

        self._output_path = output_path
        super().__init__(callbacks=callbacks, suffix=suffix)

    def save(self):
        super().save()
        self._copy_temp_path_to_output_path()

    def _copy_temp_path_to_output_path(self):
        print(f"Copying from: {self._path}")
        print(f"Copying to: {self._output_path}")
        copyfile(self._path, self._output_path)
        print("Removing tmp file...")
        Path(self._path).unlink()
        print(f"Copying done.")


def _create_writer(
    file: dict,
    output_folder: Path,
    tmp_folder: Path,
    real_spacing: float,
    shape: tuple,
) -> TmpWholeSlideMaskWriter:
    """Creates a writer

    Args:
        file (dict): dictionary containing a 'name' and 'type' key.
        output_folder (Path): folder in where output should be copied
        tmp_folder (Path): folder in where output should be kept temporary when writing
        real_spacing (float): The spacing of the output file
        shape (tuple): The shape of the ouput file

    Raises:
        ValueError: raises when type in file is not valid

    Returns:
        TmpWholeSlideMaskWriter: a writer
    """

    if file["type"] == MaskType.HEATMAP:
        callbacks = (HeatmapTileCallback(heatmap_index=file['heatmap_index']),)
    elif file["type"] == MaskType.PREDICTION:
        callbacks = (PredictionTileCallback(),)
    else:
        raise ValueError(f"Invalid file type: {file['type']}")

    writer = TmpWholeSlideMaskWriter(
        output_path=(output_folder / file["name"]), callbacks=callbacks
    )
    print(f'write: {(tmp_folder / file["name"])}')
    writer.write(
        path=(tmp_folder / file["name"]),
        spacing=real_spacing,
        dimensions=shape,
        tile_shape=(TILE_SIZE, TILE_SIZE),
    )
    return writer


def create_writers(
    image_path: Path,
    files: list,
    output_folder: Path,
    tmp_folder: Path,
) -> list:
    """Creates writers for files

    Args:
        image_path (Path): path to the images that is being processed
        files (list): files that need to be written
        output_folder (Path): folder in where output should be copied
        tmp_folder (Path): folder in where output should be kept temporary when writing

    Returns:
        list: the created writers
    """

    writers = []

    # get info
    with WholeSlideImage(image_path) as wsi:
        shape = wsi.shapes[wsi.get_level_from_spacing(SPACING)]
        real_spacing = wsi.get_real_spacing(SPACING)

    for file in files:
        if (output_folder / file["name"]).exists():
            f"Skipping prediction for {file['name']}, already exists in output folder: {output_folder}"
            continue

        writers.append(
            _create_writer(
                file=file,
                output_folder=output_folder,
                tmp_folder=tmp_folder,
                real_spacing=real_spacing,
                shape=shape,
            )
        )

    return writers
