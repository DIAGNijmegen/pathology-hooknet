from multiprocessing import Process, Queue
from .imagereader import ImageReader
from .imagewriter import ImageWriter
import numpy as np
import time
import cv2
from skimage.transform import rescale


def fit_data(data, output_shape):
    cropx = (data.shape[0] - output_shape[0]) // 2
    cropy = (data.shape[1] - output_shape[1]) // 2

    if len(data.shape) == 2:
        return data[cropx:-cropx, cropy:-cropy]
    if len(data.shape) == 3:
        return data[cropx:-cropx, cropy:-cropy, :]
    if len(data.shape) == 4:
        cropx = (data.shape[1] - output_shape[0]) // 2
        cropy = (data.shape[2] - output_shape[1]) // 2
        return data[:, cropx:-cropx, cropy:-cropy, :]
    if len(data.shape) == 5:
        cropx = (data.shape[2] - output_shape[0]) // 2
        cropy = (data.shape[3] - output_shape[1]) // 2
        return data[:, :, cropx:-cropx, cropy:-cropy, :]


class MaskReader:
    def __init__(self, mask_path):
        self._mask_path = mask_path

    @property
    def mask_path(self):
        return self._mask_path


class TifMaskReader(MaskReader):
    def __init__(self, mask_path):
        super().__init__(mask_path)


class PngMaskReader(MaskReader):
    def __init__(self, mask_path, ratio):
        super().__init__(mask_path)
        self._ratio = ratio


class ImageProcessor(Process):
    def __init__(
        self,
        wsi_path,
        mask_path,
        input_shape,
        output_shape,
        resolutions,
        command_queue,
        reader_queue,
        mask_ratio=None,
    ):

        super().__init__()
        self._wsi_path = wsi_path
        self._mask_path = mask_path
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._resolutions = resolutions
        self._command_queue = command_queue
        self._reader_queue = reader_queue

        self._wsi_ratio = 1
        self._mask_ratio = 0 if mask_ratio is None else int(mask_ratio)
        self._mask_spacing = -1
        if self._mask_path:
            if self._mask_path.endswith(".png"):
                self._mask = cv2.imread(self._mask_path, cv2.IMREAD_GRAYSCALE) / 255
            else:
                self._mask = ImageReader(self._mask_path, 0.2)

        self._wsi = ImageReader(self._wsi_path, 0.2)

        self._set_ratios()

    def run(self):
        for command_message in iter(self._command_queue.get, "STOP"):
            # decode message
            items = command_message
            # print('items', items)
            # create batch
            X_batch, masks, used_items = self._create_batch(items)
            if used_items != []:
                # put batch in de reader queue
                self._reader_queue.put((X_batch, masks, used_items))

        self._wsi.close()
        del self._wsi

    def _create_batch(self, items):
        X_batch = [[] for pixel_spacing in self._resolutions]
        used_items = []
        masks = []
        for item in items:
            col, row = item
            center_x = col + (self._output_shape[0] // 2)
            center_y = row + (self._output_shape[1] // 2)

            mask_bool, mask = self._check_mask(center_x, center_y)
            if not mask_bool:
                continue
            masks.append(mask)

            for ps_index, pixel_spacing in enumerate(self._resolutions):
                patch = self._create_patch(center_x, center_y, pixel_spacing)
                X_batch[ps_index].append(patch)
                used_items.append(item)
        return X_batch, masks, used_items

    def _set_ratios(self):
        possible_spacings = np.array([1 / (2 ** x) for x in range(4, -8, -1)])
        rounded_wsi_spacings = [
            possible_spacings[np.argmin((im_spacing - possible_spacings) ** 2)]
            for im_spacing in self._wsi.spacings
        ]
        self._wsi_ratio = self._resolutions[0] / rounded_wsi_spacings[0]
        print("WSI RATIO:", self._wsi_ratio)

        if self._mask_path and not self._mask_ratio:
            rounded_mask_spacings = [
                possible_spacings[np.argmin((im_spacing - possible_spacings) ** 2)]
                for im_spacing in self._mask.spacings
            ]
            self._mask_spacing = max(self._mask.spacings[0], self._resolutions[0])
            self._mask_ratio = rounded_mask_spacings[0] / self._resolutions[0]

        print("MASK RATIO:", self._mask_ratio)

    def _check_mask(self, centerx, centery):

        if not self._mask_path:
            return True, []

        if self._mask_path.endswith(".png"):
            # get level 0 coordinates
            mask_x = centerx * self._wsi_ratio
            mask_y = centery * self._wsi_ratio

            # center to top-left with mask ratio
            mask_x = (
                mask_x // self._mask_ratio
                - self._input_shape[0] // 2 // self._mask_ratio
            )
            mask_y = (
                mask_y // self._mask_ratio
                - self._input_shape[1] // 2 // self._mask_ratio
            )

            # get patch
            mask_patch = [
                self._mask[
                    int(mask_y) : int(mask_y)
                    + int(self._input_shape[1] // self._mask_ratio),
                    int(mask_x) : int(mask_x)
                    + int(self._input_shape[0] // self._mask_ratio),
                ]
            ]

        else:
            mask_patch = self._mask.read_center(
                centerx // self._mask_ratio,
                centery // self._mask_ratio,
                int(self._input_shape[0]),
                int(self._input_shape[1]),
                self._mask_spacing,
            )

        if 1 in np.unique(mask_patch[0]):
            # upsample
            return True, rescale(
                mask_patch[0].squeeze().astype("uint8"),
                self._mask_ratio,
                order=0,
                preserve_range=True,
            )
        return False, []

    def _create_patch(self, centerx, centery, pixel_spacing):
        patch, _, _ = self._wsi.read_center(
            centerx * self._wsi_ratio,
            centery * self._wsi_ratio,
            int(self._input_shape[0]),
            int(self._input_shape[1]),
            pixel_spacing,
        )

        return patch.astype("uint8")


class WSIReaderDeamon(Process):
    def __init__(
        self,
        wsi_path,
        mask_path,
        batch_size,
        input_shape,
        output_shape,
        tile_size,
        resolutions,
        queue_size,
        cpus,
        reader_queue,
        mask_ratio,
    ):

        super().__init__()

        # Set arguments
        self._wsi_path = wsi_path
        self._mask_path = mask_path
        self._batch_size = batch_size
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._tile_size = tile_size
        self._resolutions = resolutions
        self._queue_size = queue_size
        self._cpus = cpus
        self._reader_queue = reader_queue
        self._mask_ratio = mask_ratio

        self._pprocesses = []

        wsi = ImageReader(self._wsi_path, 0.2)
        self._wsi_y_dims, self._wsi_x_dims = wsi.shapes[wsi.level(self._resolutions[0])]
        print("DIMENSIONS")
        print(self._wsi_x_dims, self._wsi_y_dims)
        wsi.close()
        del wsi
        print("from reader deamon", self._wsi_path)
        self._minmaxes = [0, self._wsi_x_dims, 0, self._wsi_y_dims]

        # Command queue
        self._command_queue = Queue(maxsize=self._queue_size)
        self._pprocesses = []

    def _fill(self):
        print("fill")
        for y in range(self._minmaxes[2], self._minmaxes[3], self._tile_size):
            for x in range(self._minmaxes[0], self._minmaxes[1], self._tile_size):
                items = [(x, y)]
                self._command_queue.put(items)

        self.stopdeamon()

    def run(self):
        # Setup cpu processes
        for i in range(self._cpus - 2):
            self._pprocesses.append(
                ImageProcessor(
                    self._wsi_path,
                    self._mask_path,
                    self._input_shape,
                    self._output_shape,
                    self._resolutions,
                    self._command_queue,
                    self._reader_queue,
                    self._mask_ratio,
                )
            )

        # set up sampler deamon
        for i in range(self._cpus - 2):
            self._pprocesses[i].start()
        self._fill()

    def stopdeamon(self):
        for i in range(len(self._pprocesses)):
            self._command_queue.put("STOP")
        for i in range(len(self._pprocesses)):
            self._pprocesses[i].join()

        self._reader_queue.put("STOP")


class WSIWriterDeamon:
    def __init__(
        self, wsi_path, output_path, spacing, output_shape, tile_size, writer_queue
    ):

        # super().__init__()

        # Set arguments
        self._wsi_path = wsi_path
        self._output_path = output_path

        self._spacing = spacing
        self._output_shape = output_shape
        self._tile_size = tile_size
        self._writer_queue = writer_queue

        self._dtype = np.uint8
        self._coding = "monochrome"

        # get shape of image
        wsi = ImageReader(self._wsi_path, 0.2)
        self._shape = wsi.shapes[wsi.level(self._spacing)]
        writing_spacing = wsi.spacings[wsi.level(self._spacing)]
        wsi.close()
        del wsi

        # init writer
        self._writer = ImageWriter(
            image_path=self._output_path,
            shape=self._shape,
            spacing=writing_spacing,
            dtype=self._dtype,
            coding=self._coding,
            compression="lzw",
            interpolation="nearest",
            tile_size=self._tile_size,
            jpeg_quality=None,
            empty_value=0,
            skip_empty=None,
        )

        print("from writer deamon", self._wsi_path, self._output_path)

    def put(self, write_message):
        # for write_message in iter(self._writer_queue.get, 'STOP'):
        predictions, masks, items = write_message
        t1 = time.time()
        for idx, prediction in enumerate(predictions):
            col, row = items[idx]
            prediction = prediction.reshape(
                (self._output_shape[0], self._output_shape[1])
            )
            prediction = prediction[: self._tile_size, : self._tile_size].astype(
                "uint8"
            )
            mask = fit_data(masks[idx], [self._tile_size, self._tile_size]).astype(
                "uint8"
            )
            prediction *= mask

            self._writer._ImageWriter__writer.writeBaseImagePartToLocation(
                prediction.flatten(), col, row
            )
        t2 = time.time()

    def stop(self):
        self._writer.close()